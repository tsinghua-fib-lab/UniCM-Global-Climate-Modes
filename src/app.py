import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import *
from Trainer import TrainLoop
from models import UniCM
import copy
from torch.utils.tensorboard import SummaryWriter
from my_tools import norm_std, evaluation_test_data

import argparse
from numba import cuda

import setproctitle

torch.autograd.set_detect_anomaly(True)

def parse_args():
    
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for UniCM model training and evaluation")

    # --- Model & Data Parameters ---
    parser.add_argument("--model", type=str, default='UniCM', help="Model type identifier")
    parser.add_argument("--his_len", type=int, default=12, help="Length of historical input sequence (months)")
    parser.add_argument("--pred_len", type=int, default=24, help="Length of prediction sequence (months)")
    parser.add_argument("--interp", type=int, default=0, help="Interpolation steps (0 for no interpolation)")
    parser.add_argument("--climate_mode", type=str, default='all',  help="Specific climate mode for loss calculation (e.g., 'nino', 'all', 'only_0')")
    parser.add_argument("--mode_interaction", type=str, default='1', help="Flag for mode interaction settings")
    parser.add_argument("--resolution", type=int, default=5, help="Spatial resolution factor")
    parser.add_argument("--input_channal", type=int, default=1, help="Number of input data channels")
    parser.add_argument("--patch_size", type=str, default='2-2', help="Patch size as 'height-width' (e.g., '5-5')")
    parser.add_argument("--time_patch", type=int, default=0, help="Flag for using temporal patching")
    parser.add_argument("--t_patch_len", type=int, default=1, help="Length of temporal patches")
    parser.add_argument("--stride", type=int, default=1, help="Stride for patching")
    parser.add_argument("--training_data", type=str, default='CESM2-FV2*gr', help="Identifier for training dataset")
    parser.add_argument("norm_std", type=int, default=1, help="Normalization strategy flag (1 for std normalization)")
    parser.add_argument("--t20d_mode", type=int, default=1, help="Flag for including T20D data channel")

    # --- Transformer Architecture Parameters ---
    parser.add_argument("--d_size", type=int, default=256, help="Embedding dimension (d_model)")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of the feedforward network (default: d_size * 4)")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # --- Training Parameters ---
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay (L2 regularization)")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate for annealing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--early_stop", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--lr_early_stop", type=int, default=5, help="Patience for learning rate scheduler early stop")

    # --- Loss Coefficients ---
    parser.add_argument("--vdt_coef", type=float, default=1.0, help="Coefficient for the main reconstruction loss")
    parser.add_argument("--mode_coef", type=float, default=1.0, help="Coefficient for the climate mode skill loss")
    parser.add_argument("--ours_coef", type=float, default=0.0, help="Coefficient for the explicit mode prediction loss")
    parser.add_argument("--loss_nino", type=int, default=1, help="Flag to include specific Nino loss")
    parser.add_argument("--loss_all", type=int, default=0, help="Flag for alternative loss calculation")

    # --- Environment & IO ---
    parser.add_argument("--mode", type=str, default='training', choices=['testing','finetuning','training', 'training_finetuning'])
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--pretrain_model", type=str, default='None', help="Path to a pretrained model to load")
    parser.add_argument("--pretrained_simulate_path", type=str, default='', help="Path to a pretrained simulation model")
    parser.add_argument("--log_interval", type=int, default=20, help="Step interval for logging training loss")
    parser.add_argument("--val_interval", type=int, default=1, help="Epoch interval for running validation")
    parser.add_argument("--machine", type=str, default='LM1', help="Machine identifier")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    
    # --- Miscellaneous ---
    parser.add_argument("--norm_std", type=int, default=1, help="Flag for normalization strategy")
    parser.add_argument("--explore", type=int, default=1, help="Exploration flag")
    parser.add_argument("--sv_ratio", type=float, default=0.0, help="SV ratio")
    parser.add_argument("--autoregressive", type=int, default=0, help="Flag for autoregressive prediction")
    parser.add_argument("--num_bagging", type=int, default=5, help="Number of bagging runs")

    args = parser.parse_args()
    return args


def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(mypara):
    """
    device and path
    """

    setproctitle.setproctitle("@yuanyuan")

    device = torch.device(f"cuda:{mypara.cuda_id}" if torch.cuda.is_available() else "cpu")
    mypara.device = device

    file_path = 'SaveModel_Seed{}_{}/'.format(mypara.seed, mypara.machine)

    print('file path: {}'.format(file_path), 'input_channal: {}'.format(mypara.input_channal))

    mypara.model_path = './experiments/{}'.format(file_path)
    writer = SummaryWriter(log_dir = './logs/{}'.format(file_path))

    if not os.path.exists('./experiments/'):
        os.makedirs('./experiments/')

    if not os.path.exists(mypara.model_path):
        os.makedirs(mypara.model_path)
        os.makedirs(mypara.model_path+'model_save/')

    mypara.train_scaler = {}
    mypara.val_scaler = {}

    mypara.patch_size = [int(i) for i in mypara.patch_size.split('-')]
    mypara.emb_spatial_size = 12 * 72 // (mypara.patch_size[0] * mypara.patch_size[1])

    mypara.template = []

    if 'training' in mypara.mode:
        """
        data preparation
        """
        
        train_data = make_data(mypara).dataloader_seq('tos_Omon_{}_historical_r1i1p1f1_{}_185001-201412.nc'.format(mypara.training_data.split('*')[0],mypara.training_data.split('*')[1]))
        train_data_tauu = make_data(mypara).dataloader_seq('tauu_Amon_CESM2-FV2_historical_r1i1p1f1_gn_1850_2014.nc', 'tauu')
        train_data_tauv = make_data(mypara).dataloader_seq('tauv_Amon_CESM2-FV2_historical_r1i1p1f1_gn_1850_2014.nc', 'tauv')
        train_data_t20d = make_data(mypara).dataloader_seq('t20d_Emon_EC-Earth3-Veg-LR_historical_r1i1p1f1_gn_195001-201412.nc','t20d')
        train_data_thetao = make_data(mypara).dataloader_seq('thetaot300_Emon_EC-Earth3-CC_historical_r1i1p1f1_gn_1850_2014.nc', 'thetaot300')
        train_data = [torch.cat([train_data[0], train_data_tauu[0], train_data_tauv[0], train_data_thetao[0],train_data_t20d[0]],dim=2),train_data[1],train_data[2],train_data[3]]
        val_data = make_val_data_ORAS5(mypara).dataloader_seq()
        train_data = torch.utils.data.TensorDataset(*train_data)
        val_data = torch.utils.data.TensorDataset(*val_data)
        val_data = torch.utils.data.DataLoader(val_data, num_workers=4, batch_size=mypara.batch_size*4, shuffle=False) 
        train_data = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=mypara.batch_size, shuffle=True) 

        """
        model initialization
        """

        mymodel = UniCM(mypara).to(device)

        if mypara.pretrained_simulate_path!='':

            pretrained_path = './experiments/{}_Seed{}_LM2/model_save/model_best.pkl'.format(mypara.pretrained_simulate_path, mypara.seed)
            mymodel.load_state_dict(torch.load(pretrained_path),strict=True)
            print('load trained model from simulation data')

        else:
            print('no pretrained model, train from scratch')

        model_size = sum(p.numel() * p.element_size() for p in mymodel.parameters()) 
        model_size_mb = model_size / (1024 ** 2)

        print(f"\nmodel size: {model_size_mb:.2f} MB\n")

        """
        model pretraining
        """

        # trianing
        TrainLoop(
            args = mypara,
            writer = writer,
            model=mymodel,
            train_data= train_data,
            val_data = val_data,
            device=device,
        ).run_loop()

   
    """
    model evaluation
    """

    if mypara.mode in ['testing','finetuning','training', 'training_finetuning']:

        mypara.feature_index = [0]

        mypara.mode = 'testing'

        mypara.batch_size  = mypara.batch_size // 2

        test_data = make_test_data_ORAS5(mypara).dataloader_seq()

        mymodel = Geoformer(mypara).to(device)

        # load pretrained best model from the last training
        if mypara.pretrained_path == '':
            pretrained_path = './experiments/{}model_save/model_best.pkl'.format(file_path)
        else:
            pretrained_path = './experiments/{}/model_save/model_best.pkl'.format(mypara.pretrained_path)

        if os.path.exists(pretrained_path):
            mymodel.load_state_dict(torch.load(pretrained_path),strict=False)
            print('load finetuned model from:', pretrained_path)
        else:
            print('no pretrained model, train from scratch')

        if mypara.pretrained_path == '':
            mypara.model_path = './experiments/testing_{}'.format(file_path)
            writer = SummaryWriter(log_dir = './logs/testing_{}'.format(file_path))
        else:
            mypara.model_path = './experiments/testing_NBag{}_{}/'.format(mypara.num_bagging, mypara.pretrained_path)
            writer = SummaryWriter(log_dir = './logs/testing_NBag{}_{}/'.format(mypara.num_bagging, mypara.pretrained_path))
            
        
        if not os.path.exists(mypara.model_path):
            os.makedirs(mypara.model_path)

        if not os.path.exists(mypara.model_path+'model_save/'):
            os.makedirs(mypara.model_path+'model_save/')

        print(mypara.model_path)

        print('-------------------------------\n')
            
        with open(mypara.model_path+'result_all.txt', 'a') as f:
            f.write('\n-----------------------------------------------------OursRefine: {}-----------------------------------------------------\n'.format(mypara.ours_refine ))
            f.close()

        print('testing on ERA5 dataset')
        with open(mypara.model_path+'result_all.txt', 'a') as f:
            f.write('\n--------------------------------dataset:ERA5--------------------------------\n')
            f.close()
        test_data = make_test_data_ERA5(mypara).dataloader_seq()
        print(mymodel.mypara.val_relative)
        test_data_assist = make_test_data_ORAS5(mypara).dataloader_seq()
        test_data = [torch.cat((test_data[0], test_data_assist[0][:,:,-2:]), dim = 2), test_data[1], test_data[2], test_data[3]]
        mypara.current_test_data = 'ERA5'
        evaluation_test_data(test_data, mypara, writer, mymodel, device)

        print('testing on ORAS5 dataset')
        with open(mypara.model_path+'result_all.txt', 'a') as f:
            f.write('\n--------------------------------dataset:ORAS5--------------------------------\n')
            f.close()
        test_data = make_test_data_ORAS5(mypara).dataloader_seq()
        print(mymodel.mypara.val_relative)
        mypara.current_test_data = 'ORAS5'
        evaluation_test_data(test_data, mypara, writer, mymodel, device)

        print('testing on SODA dataset')
        with open(mypara.model_path+'result_all.txt', 'a') as f:
            f.write('\n--------------------------------dataset:SODA--------------------------------\n')
            f.close()
        test_data = make_test_data_SODA224(mypara).dataloader_seq()
        test_data_hc = make_test_data_SODA224(mypara).dataloader_seq(name='hc')
        test_data_assist = make_test_data_ORAS5(mypara).dataloader_seq()

        print(test_data[0].shape, test_data_assist[0].shape, test_data_hc[0].shape)

        test_data = [torch.cat((test_data[0], test_data_assist[0][:test_data[0].shape[0],:,1:3], test_data_hc[0], test_data_assist[0][:test_data[0].shape[0],:,-1:]), dim = 2), test_data[1], test_data[2], test_data[3]]

        mypara.current_test_data = 'SODA'
        evaluation_test_data(test_data, mypara, writer, mymodel, device)

        print('testing on GODAS dataset')
        with open(mypara.model_path+'result_all.txt', 'a') as f:
            f.write('\n--------------------------------dataset:GODAS--------------------------------\n')
            f.close()
        test_data = make_test_data_GODAS(mypara).dataloader_seq(name='tos')
        test_data_hc = make_test_data_GODAS(mypara).dataloader_seq(name='hc')
        test_data_assist = make_test_data_ORAS5(mypara).dataloader_seq()

        test_data = [torch.cat((test_data[0][:test_data_assist[0].shape[0]], test_data_assist[0][:,:,1:3], test_data_hc[0][:test_data_assist[0].shape[0]], test_data_assist[0][:,:,-1:]), dim = 2), test_data[1][:test_data_assist[0].shape[0]], test_data[2][:test_data_assist[0].shape[0]], test_data[3][:test_data_assist[0].shape[0]]]
        mypara.current_test_data = 'GODAS'
        evaluation_test_data(test_data, mypara, writer, mymodel, device)



if __name__ == "__main__":

    mypara = parse_args()
    setup_init(mypara.seed)
    main(mypara)