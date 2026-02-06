"""
Trainer.py

This module defines the main training and evaluation loop (TrainLoop) for the climate model.
It includes functionalities for:
- Forward and backward passes
- Loss calculation (including climate mode skill scores)
- Learning rate scheduling (warmup and annealing)
- Validation and testing loops
- GPU usage monitoring
"""

import torch
from torch.optim import AdamW
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import os
import pynvml

def unnormalize(x, MAX, MIN):
    return (x * (MAX-MIN)/2 + (MAX+MIN)/2).clamp(MIN, MAX)


def SMAPE(pred, target):
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    t = target[target!=0]
    p = pred[target!=0]
    return torch.sum(2.0 * (t - p).abs() / (t.abs() + p.abs())).item(), target[target!=0].numel()


class TrainLoop:
    def __init__(self, args, model, train_data=None, val_data=None, train_index=None, val_index=None, device=None):
        self.args = args
        self.model = model
        self.train_data = train_data
        self.train_index = train_index
        self.val_data = val_data
        self.val_index = val_index
        self.device = device
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=self.weight_decay)
        self.log_interval = args.log_interval
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_rmse = 1e9
        self.best_sc = -1e9
        self.early_stop = 0
        self.lr_anneal_steps = args.lr_anneal_steps
        ninoweight = np.array([1] * 10 + [1.5] * 10 + [1] * (args.pred_len-20)) * np.log(np.arange(args.pred_len) + 1.2)
        self.ninoweight = ninoweight[: self.args.pred_len]
        self.flag = 0
        self.sv_ratio = args.sv_ratio
        self.special_index = [4,5] if len(args.val_relative)>2 else [1]

    def pearson_corr(self, x, y):
        # Center the data
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        
        # Calculate covariance and standard deviation
        cov = torch.sum(x * y) / len(x)
        std_x = torch.sqrt(torch.sum(x ** 2) / len(x))
        std_y = torch.sqrt(torch.sum(y ** 2) / len(y))
        
        # Calculate and return Pearson correlation coefficient
        return cov / (std_x * std_y)
    def calscore(self, y_pred, y_true):
        # compute Nino score
        y_pred_avg = torch.zeros_like(y_pred)
        y_true_avg = torch.zeros_like(y_true)

        y_pred_avg[:, 1:-1] = (y_pred[:, :-2] + y_pred[:, 1:-1] + y_pred[:, 2:]) / 3
        y_true_avg[:, 1:-1] = (y_true[:, :-2] + y_true[:, 1:-1] + y_true[:, 2:]) / 3
        y_pred_avg[:, 0] = y_pred[:, 0]
        y_true_avg[:, 0] = y_true[:, 0]
        y_pred_avg[:, -1] = y_pred[:, -1] 
        y_true_avg[:, -1] = y_true[:, -1]

        pearson_corr_all = np.zeros(self.args.pred_len)

        for lead in range(self.args.pred_len):
            A = y_pred_avg[:, lead]
            B =  y_true_avg[:, lead]
            pearson_corr = self.pearson_corr(A, B)
            pearson_corr_all[lead] = pearson_corr.item()

        return pearson_corr_all
    
    def get_teacher_forcing_ratio(self, initial_ratio=1.0, final_ratio=0.0):
        if self.step <= 200:
            ratio = initial_ratio
        elif self.step > 200 and self.step <= self.args.lr_anneal_steps-500:
            total_decay_steps = self.args.lr_anneal_steps - 700  # Actual steps involved in decay
            decay_step = self.step - 200  # Current decay step
            ratio = initial_ratio - (initial_ratio - final_ratio) * (decay_step / total_decay_steps)
        else:
            ratio = final_ratio
        return max(final_ratio, ratio)

    def loss_var(self, y_pred, y_true):
        rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
        rmse = rmse.sqrt().mean(dim=0)
        rmse = torch.sum(rmse, dim=[0, 1])
        return rmse

    def loss_nino(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()

    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss

    def climate_mode_extract(self, input, scaler=None):

        output_all = []
        for mode in range(len(self.args.val_relative)):
            output = self.data2mode(mode, input, input, self.args.val_relative[mode], scaler, self.special_index)[0]

            output_all.append(output.unsqueeze(dim=1))

        return torch.cat(output_all,dim=1)

    def climate_WWV_extract(self, input, scaler=None):

        output_all = []

        output = self.data2mode(0, input, input, self.args.WWV, scaler, self.special_index)[0]
        output_all.append(output.unsqueeze(dim=1))

        return torch.cat(output_all,dim=1)


    def climate_mode_sc(self, y_pred_all, y_target_all, scaler, mode='testing', refine_data=None, save=False):

        pred, target = [], []
        for index in range(len(self.args.val_relative)):
            eino_pred, eino_target = self.data2mode(index, y_pred_all, y_target_all, self.args.val_relative[index], scaler, self.special_index)

            pred.append(eino_pred)
            target.append(eino_target)

        if len(pred)>1:
            pred = torch.stack(pred, dim=1)
            target = torch.stack(target, dim=1)

        else:
            pred = pred.unsqueeze(dim=1)
            target = target.unsqueeze(dim=1)
            refine_data = refine_data.unsqueeze(dim=1)

        if save:
            torch.save(pred, self.args.model_path+'pred_{}.pkl'.format(self.args.current_test_data))
            torch.save(target, self.args.model_path+'target_{}.pkl'.format(self.args.current_test_data))

        if mode=='testing':

            sc_all = [self.calscore(pred[:,i],target[:,i]) for i in range(pred.shape[1])]

            sc_mean = [np.mean(sc) for sc in sc_all]

            if self.args.metrics_mode == 'weight':
                sc = [np.mean(i*self.ninoweight) for i in sc_all]
            else:
                sc = sc_mean

            result = {'sc_all':[list(i) for i in sc_all], 'sc':sc[0], 'sc_avg':sc_mean}

        else:
            loss = torch.mean((pred-target)**2)
            result = {'loss':loss}

        return result


    def Evaluation_ensemble(self, epoch):
        with torch.no_grad():

            fine_pred = []
            fine_target = []

            coarse_pred = []
            coarse_target = []

            physical_field_pred = []
            physical_field_target = []

            for path in ['experiments/'+self.args.pretrained_path +'_Seed{}/model_save/model_best.pkl'.format(i) for i in range(1,self.args.num_ensemble+1)]:
                self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=True)
                self.model.to(self.device)
                print('load model from {}'.format(path))

                self.model.eval()
                
                eino_pred = []
                eino_target = []

                mode_all_pred = []
                mode_all_target = []

                index_all = []

                timestamps_all = []

                field_pred = []
                field_target = []

                for batch in self.val_data:
                    value, timestamps, std = batch

                    mode_align_value = self.climate_mode_extract(value[:,:,0].to(self.device))

                    assert value.shape[2]==5
                    mode_WWV = self.climate_WWV_extract(value[:,:,-1].to(self.device))
                    mode_align_value = torch.cat((mode_align_value,mode_WWV),dim=1)

                    value = value[:,:,:self.args.input_channal]

                    batch = (value, timestamps, None, None, mode_align_value)

                    pred, target, pred_mode, target_mode = self.model_forward(batch, self.model, mode='testing') 

                    field_pred.append(pred.clone()[:,:,0])
                    field_target.append(target.clone()[:,:,0])

                    pred = pred[:,:,0]
                    target = target[:,:,0]

                    pred *= std.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)
                    target *= std.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)

                    eino_pred.append(pred)
                    eino_target.append(target)

                    mode_all_pred.append(pred_mode)
                    mode_all_target.append(target_mode)
                    timestamps_all.append(timestamps)
                    
                    
                eino_pred = torch.cat(eino_pred, dim=0)
                eino_target = torch.cat(eino_target, dim=0)

                mode_all_pred = torch.cat(mode_all_pred, dim=0)
                mode_all_target = torch.cat(mode_all_target, dim=0)
                timestamps_all = torch.cat(timestamps_all, dim=0)

                field_pred = torch.cat(field_pred, dim=0)
                field_target = torch.cat(field_target, dim=0)
 
                fine_pred.append(eino_pred)
                fine_target.append(eino_target)
                coarse_pred.append(mode_all_pred)
                coarse_target.append(mode_all_target)

                physical_field_pred.append(field_pred)
                physical_field_target.append(field_target)

            fine_pred = torch.stack(fine_pred, dim=0)
            fine_target = torch.stack(fine_target, dim=0)
            coarse_pred = torch.stack(coarse_pred, dim=0)
            coarse_target = torch.stack(coarse_target, dim=0)

            eino_pred = torch.mean(fine_pred, dim=0)
            eino_target = torch.mean(fine_target, dim=0)
            mode_all_pred = torch.mean(coarse_pred, dim=0)
            mode_all_target = torch.mean(coarse_target, dim=0)

            physical_field_pred = torch.mean(torch.stack(physical_field_pred, dim=0), dim=0)
            physical_field_target = torch.mean(torch.stack(physical_field_target, dim=0), dim=0)

            print('save file size: ', mode_all_pred.shape, mode_all_target.shape)

        torch.save(physical_field_pred, self.args.model_path+'physical_field_pred_{}.pkl'.format(self.args.current_test_data))
        torch.save(physical_field_target, self.args.model_path+'physical_field_target_{}.pkl'.format(self.args.current_test_data))

        # # save data
        torch.save(timestamps_all, self.args.model_path+'timestamps_all_{}.pkl'.format(self.args.current_test_data))

        result = self.climate_mode_sc(eino_pred, eino_target, None,  mode='testing',save=True)
        
        sc_all = result['sc_all']
        sc = result['sc']
        sc_avg = result['sc_avg']

        rmse = torch.sqrt(torch.mean((eino_pred - eino_target) ** 2)).item()
            
        self.best_sc = sc 
        self.best_sc_finegrained = sc_all
        self.best_sc_avg = sc_avg
        self.early_stop = 0
        self.best_rmse = rmse
        print('epoch:{}, SC_best:{}, SC_avg:{}'.format(epoch, self.best_sc, sc_avg))
        print('model saved')
        with open(self.args.model_path+self.args.result_filename, 'a') as f:
            f.write('\n----------evaluation approach------------')
            f.write('\nepoch: {}'.format(epoch))
            f.write('\nSC: {}'.format(self.best_sc))
            f.write('\nSC_avg: {}'.format(sc_avg))
            f.write('\nRMSE: {}'.format(rmse))
            f.write('\nfine-grained SC: {}\n'.format([list(i) for i in self.best_sc_finegrained]))
            f.close()
    
    def Evaluation(self, epoch):

        self.model.eval()
        
        num_all, loss_real_all = 0.0, 0.0
        start = time.time()

        with torch.no_grad():

            eino_pred = []
            eino_target = []

            mode_all_pred = []
            mode_all_target = []

            for batch in self.val_data:
                value, timestamps, std = batch

                mode_align_value = self.climate_mode_extract(value[:,:,0].to(self.device))

                assert value.shape[2]==5
                mode_WWV = self.climate_WWV_extract(value[:,:,-1].to(self.device))
                mode_align_value = torch.cat((mode_align_value,mode_WWV),dim=1)

                value = value[:,:,:self.args.input_channal]

                batch = (value, timestamps, None, None, mode_align_value)

                pred, target, pred_mode, target_mode = self.model_forward(batch, self.model, mode='testing') 

                pred = pred[:,:,0]
                target = target[:,:,0]

                pred *= std.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)
                target *= std.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).to(self.device)

                eino_pred.append(pred)
                eino_target.append(target)

                mode_all_pred.append(pred_mode)
                mode_all_target.append(target_mode)

                num_all += target.numel()
                loss_real_all += torch.mean(((pred-target)**2).detach().cpu()) * target.numel()

        rmse_val = np.sqrt(loss_real_all / num_all)

        eino_pred = torch.cat(eino_pred, dim=0)
        eino_target = torch.cat(eino_target, dim=0)

        mode_all_pred = torch.cat(mode_all_pred, dim=0)
        mode_all_target = torch.cat(mode_all_target, dim=0)

    
        result = self.climate_mode_sc(eino_pred, eino_target, None, mode='testing', refine_data=mode_all_pred)
        sc_all = result['sc_all']
        sc = result['sc']
        sc_avg = result['sc_avg']

        if sc > self.best_sc:
            self.best_sc = sc 
            self.best_sc_finegrained = sc_all
            self.best_sc_avg = sc_avg
            self.early_stop = 0
            self.best_rmse = rmse_val
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_itr_{}.pkl'.format(epoch))
            print('epoch:{}, SC_best:{}, SC_avg:{}'.format(epoch, self.best_sc, sc_avg))
            # print([list(i) for i in self.best_sc_finegrained])
            print('model saved')
            with open(self.args.model_path+self.args.result_filename, 'a') as f:
                f.write('\n----------evaluation approach------------')
                f.write('\nepoch: {}'.format(epoch))
                f.write('\nSC: {}'.format(self.best_sc))
                f.write('\nSC_avg: {}'.format(sc_avg))
                f.write('\nRMSE: {}'.format(rmse_val))
                f.write('\nfine-grained SC: {}\n'.format([list(i) for i in self.best_sc_finegrained]))
                f.close()

        else:
            if epoch > 1:
                self.early_stop += 1
            if self.early_stop % self.args.lr_early_stop ==0 and self.early_stop > 0:
                for param_group in self.opt.param_groups:
                    current_lr = param_group["lr"]
                    break
                if current_lr > self.args.min_lr: 
                    new_lr = max(current_lr * 0.5, self.args.min_lr)
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = new_lr
                    #self.early_stop = 0
                    if self.args.explore==0:
                        self.model.load_state_dict(torch.load(self.args.model_path+'model_save/model_best.pkl'))
                    print('Learning rate reduced to {}'.format(new_lr))

            print('epoch:{}, sc:{}, sc_best:{}, early_stop:{}\n'.format(epoch, sc, self.best_sc, self.early_stop))
            if self.early_stop >= self.args.early_stop:
                print('Early stop!')
                with open(self.args.model_path+self.args.result_filename, 'a') as f:
                    f.write('\n\n---------------final result----------------')
                    f.write('\nepoch:{}, SC:{}, SC_avg:{}, RMSE:{}'.format(epoch, self.best_sc, self.best_sc_avg, self.best_rmse))
                    f.write('\nfine-grained sc: {}'.format([list(i) for i in self.best_sc_finegrained]))
                    f.close()
                return 'stop'

        return 'continue'



    def run_loop(self):
        step = 0
        iteration = 0
        if self.args.mode == 'testing':
            if self.args.pretrained_path=='':
                state = self.Evaluation(0)
            else:
                state = self.Evaluation_ensemble(0)
            return 0

        print('Evaluation First!')
        state = self.Evaluation(0)

        for epoch in range(self.args.epochs):
            print('Training')
            
            loss_all, num_all, loss_real_all = 0.0, 0.0,0.0
            start = time.time()

            for index, batch in enumerate(self.train_data):
                value, timestamps, std = batch

                self.step = iteration + 1
                
                mode_align_value = self.climate_mode_extract(value[:,:,0].to(self.device))
                
                assert value.shape[2]==5
                mode_WWV = self.climate_WWV_extract(value[:,:,-1].to(self.device))
                mode_align_value = torch.cat((mode_align_value,mode_WWV),dim=1)

                value = value[:,:,:self.args.input_channal]

                batch = (value, timestamps, None, None, mode_align_value)

                self.model.train()
                loss, loss_real, num  = self.run_step(batch, step)
                step += 1
                loss_all += loss * num
                loss_real_all += loss_real * num
                num_all += num

            print('\nEvaluation with eval')
            state = self.Evaluation(epoch)

            if state == 'stop':
                return 0

            end = time.time()

            print('training time:{} min'.format(round((end-start)/60.0,2)))
            print('epoch:{}, training loss:{}, training rmse:{}'.format(epoch, loss_all / num_all, np.sqrt(loss_real_all / num_all)))

    def model_forward(self, batch, model, data=None, mode='training'):

        predictor = batch[0].to(self.device).float()
        timestamps = batch[1].to(self.device).int()
        predictor_align = batch[4].to(self.device).float()

        assert torch.isfinite(predictor).all(), "Input contains NaN or Inf"
        assert torch.isfinite(predictor_align).all(), "Input contains NaN or Inf"
        assert torch.isfinite(timestamps).all(), "Input time contains NaN or Inf"

        if model.training:
            self.sv_ratio = max(self.sv_ratio - 1e-3, 0)

        pred, pred_mode = self.model(
                predictor,
                predictor_align,
                timestamps,
                train = model.training,
                sv_ratio = self.sv_ratio,
            )

        target = predictor[:, -self.args.pred_len:]

        target_mode = predictor_align[:, :, -self.args.pred_len:]

        return pred, target, pred_mode, target_mode


    def data2mode(self,index, pred, target, relative_index, scaler, special_index= [4,5]):

        if index not in special_index:
            pred = pred[:,:, relative_index[0]:relative_index[1], relative_index[2]:relative_index[3]]
            target = target[:,:, relative_index[0]:relative_index[1], relative_index[2]:relative_index[3]]
            pred = torch.mean(pred, axis=(2,3))
            target = torch.mean(target, axis=(2,3))

        else:
            pred1 = pred[:,:, relative_index[0][0]:relative_index[0][1], relative_index[0][2]:relative_index[0][3]]
            target1 = target[:,:, relative_index[0][0]:relative_index[0][1], relative_index[0][2]:relative_index[0][3]]
            pred2 = pred[:,:, relative_index[1][0]:relative_index[1][1], relative_index[1][2]:relative_index[1][3]]
            target2 = target[:,:, relative_index[1][0]:relative_index[1][1], relative_index[1][2]:relative_index[1][3]]
            pred = torch.mean(pred1, axis=(2,3)) - torch.mean(pred2, axis=(2,3))
            target = torch.mean(target1, axis=(2,3)) - torch.mean(target2, axis=(2,3))

        return pred, target

    def forward_backward(self, batch, step):

        pred, target, pred_mode, target_mode = self.model_forward(batch, self.model)

        if self.args.loss_all == 1:
            loss = torch.mean((pred - target)**2)

        else:
            loss = torch.mean((pred[:,:,0] - target[:,:,0])**2)

        pred = pred[:,:,0]
        target = target[:,:,0]

        loss2 = self.climate_mode_sc(pred, target, None, mode='training')['loss']

        mode_loss = torch.mean((pred_mode - target_mode) ** 2)

        loss = loss * self.args.lambda1 + loss2 * self.args.lambda2 + mode_loss * self.args.lambda3

        loss.backward()
        num = target.numel()
        loss_real = torch.mean(((pred-target)**2).detach().cpu())
        
        return loss.item(), loss_real, num

    def run_step(self, batch, step):
        self.opt.zero_grad()
        loss, loss_real, num = self.forward_backward(batch, step)
        self._anneal_lr()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
        self.opt.step()
        return loss, loss_real, num

    def _anneal_lr(self):
        if self.step < self.warmup_steps:
            lr = self.lr * (self.step+1) / self.warmup_steps
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
