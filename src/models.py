import torch
import torch.nn as nn
from my_tools import make_embedding, unfold_func, miniEncoder, miniDecoder, fold_func

class UniCM(nn.Module):
    def __init__(self, mypara):
        super().__init__()
        self.mypara = mypara
        d_size = mypara.d_size
        self.device = mypara.device
        self.d_size = d_size
        self.cube_dim = mypara.input_channal * mypara.patch_size[0] * mypara.patch_size[1]
        self.predictor_emb = make_embedding(
            cube_dim=self.cube_dim,
            d_size=d_size,
            emb_spatial_size=mypara.emb_spatial_size,
            device=self.device,
        )
        self.predictand_emb = self.predictor_emb

        enc_layer = miniEncoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        dec_layer = miniDecoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout
        )
        self.encoder = multi_enc_layer(
            enc_layer=enc_layer, num_layers=mypara.num_encoder_layers
        )
        self.decoder = multi_dec_layer(
            dec_layer=dec_layer, num_layers=mypara.num_decoder_layers
        )
        self.linear_output = nn.Linear(d_size, self.cube_dim)

        self.predictor_emb_mode = make_embedding(
            cube_dim=1,
            d_size=d_size,
            emb_spatial_size=len(mypara.val_relative) + mypara.t20d_mode,
            device=self.device,
        )
        
        self.predictor_emb_mode.time_emb = self.predictor_emb.time_emb
        self.predictand_emb_mode = self.predictor_emb_mode

        self.linear_output_mode = nn.Linear(d_size, 1)

        self.enc_layer_prompt = miniEncoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout, mode='T'
        )
        self.encoder_prompt = multi_enc_layer(
            enc_layer=self.enc_layer_prompt, num_layers=mypara.num_encoder_layers
        )
        self.prompt_linear = nn.Linear(1, d_size)
        
        enc_layer_mode = miniEncoder(
        d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout, mode = 'ST' if mypara.mode_interaction !='0' else 'T'
        )

        dec_layer_mode = miniDecoder(
            d_size, mypara.nheads, mypara.dim_feedforward, mypara.dropout, mode = 'ST' if mypara.mode_interaction !='0' else 'T'
        )

        self.encoder_mode = multi_enc_layer(
            enc_layer=enc_layer_mode, num_layers=mypara.num_encoder_layers
        )

        self.decoder_mode = multi_dec_layer(
            dec_layer=dec_layer_mode, num_layers=mypara.num_decoder_layers
        )

        self.special_index = [4,5] if len(mypara.val_relative)>2 else [1]
        self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    

    def forward_sep(
        self,
        predictor,
        timestamps,
        encoder,
        decoder,
        linear_output,
        predictor_emb,
        predictand_emb,
        cube_dim,
        patch_size,
        in_mask=None,
        enout_mask=None,
        train=True,
        sv_ratio=0,
        bias_enc = None,
        bias_dec = None,
    ):
        """
        mypara:
            predictor: (batch, lb, C, H, W)
            predictand: (batch, pre_len, C, H, W)
        Returns:
            outvar_pred: (batch, pre_len, C, H, W)
        """
        
        predictor, predictand = predictor[:,:self.mypara.his_len], predictor[:,self.mypara.his_len:]
        predictor_time, predictand_time = timestamps[:,:self.mypara.his_len], timestamps[:,self.mypara.his_len:]

        en_out = self.encode(encoder=encoder, predictor=predictor, timestamps = predictor_time, cube_dim = cube_dim, in_mask=in_mask, predictor_emb=predictor_emb, patch_size = patch_size, bias_enc=bias_enc)
        if train:
            if self.mypara.autoregressive == 0:
                with torch.no_grad():
                    connect_inout = torch.cat(
                        [predictor[:, -1:], predictand[:, :-1]], dim=1
                    )
                    time = torch.cat([predictor_time[:, -1:], predictand_time[:, :-1]], dim=1)
                    out_mask = self.make_mask_matrix(connect_inout.size(1))
                    outvar_pred, outvar_pred_emb = self.decode(
                        decoder,
                        linear_output,
                        connect_inout,
                        time,
                        cube_dim,
                        en_out,
                        out_mask,
                        enout_mask,
                        predictand_emb = predictand_emb,
                        patch_size = patch_size,
                        bias_dec=bias_dec,
                    )
            else:
                connect_inout = torch.cat(
                    [predictor[:, -1:], predictand[:, :-1]], dim=1
                )
                time = torch.cat([predictor_time[:, -1:], predictand_time[:, :-1]], dim=1)
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred, outvar_pred_emb = self.decode(
                    decoder,
                    linear_output,
                    connect_inout,
                    time,
                    cube_dim,
                    en_out,
                    out_mask,
                    enout_mask,
                    predictand_emb = predictand_emb,
                    patch_size = patch_size,
                    bias_dec=bias_dec,
                )
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio
                    * torch.ones(predictand.size(0), predictand.size(1) - 1, 1, 1, 1)
                ).to(self.device)
            else:
                supervise_mask = 0

            predictand = (
                supervise_mask * predictand[:, :-1]
                + (1 - supervise_mask) * outvar_pred[:, :-1]
            )
            predictand = torch.cat([predictor[:, -1:], predictand], dim=1)
            time = torch.cat([predictor_time[:, -1:], predictand_time[:,:-1]], dim=1)
            # predicting
            outvar_pred, outvar_pred_emb = self.decode(
                decoder,
                linear_output,
                predictand,
                time,
                cube_dim,
                en_out,
                out_mask,
                enout_mask,
                predictand_emb = predictand_emb,
                patch_size = patch_size,
                bias_dec=bias_dec,
            )

        else:
            predictand = None
            assert predictand is None
            predictand = predictor[:, -1:]
            time = predictor_time[:, -1:]
            for t in range(self.mypara.pred_len):
                out_mask = self.make_mask_matrix(predictand.size(1))
                outvar_pred, outvar_pred_emb = self.decode(
                    decoder,
                    linear_output,
                    predictand,
                    time,
                    cube_dim,
                    en_out,
                    out_mask,
                    enout_mask,
                    predictand_emb = predictand_emb,
                    patch_size = patch_size,
                    bias_dec=bias_dec,
                )
                predictand = torch.cat([predictand, outvar_pred[:, -1:]], dim=1)
                time = torch.cat([time, predictand_time[:, t : t+1]], dim=1)

        return outvar_pred, en_out, outvar_pred_emb

    def forward(
        self,
        predictor, # B * T * C * H * W
        predictor_mode, # B * 7 * T
        timestamps, # B * T
        index_time, 
        in_mask=None,
        enout_mask=None,
        train=True,
        sv_ratio=0,
    ):

        B, T, C, H, W = predictor.shape

        predictor_mode = predictor_mode.permute(0,2,1).unsqueeze(-1).unsqueeze(2)

        pred_mode, enc_out_mode, outvar_pred_emb_mode = self.forward_sep(predictor_mode, timestamps, self.encoder_mode, self.decoder_mode, self.linear_output_mode, self.predictor_emb_mode, self.predictand_emb_mode, 1, [1,1], in_mask, enout_mask, train, sv_ratio)

        enc_out = torch.zeros(B, H//self.mypara.patch_size[0], W//self.mypara.patch_size[1], self.mypara.his_len, enc_out_mode.shape[-1]).to(self.device)
        outvar_pred_emb = torch.zeros(B, H//self.mypara.patch_size[0], W//self.mypara.patch_size[1], outvar_pred_emb_mode.shape[-2], outvar_pred_emb_mode.shape[-1]).to(self.device)

        for index in range(len(self.mypara.val_relative)):
            
            if index not in self.special_index:
                
                start1 = self.mypara.val_relative[index][0]//self.mypara.patch_size[0]
                end1 = self.mypara.val_relative[index][1]//self.mypara.patch_size[0]
                start2 = self.mypara.val_relative[index][2]//self.mypara.patch_size[1]
                end2 = self.mypara.val_relative[index][3]//self.mypara.patch_size[1]
                if start1==end1:
                    end1 = end1+1

                if start2==end2:
                    end2 = end2+1

                enc_out[:,start1:end1,start2:end2] = enc_out[:,start1:end1,start2:end2] + enc_out_mode[:,index:index+1].unsqueeze(1)
                outvar_pred_emb[:,start1:end1,start2:end2] = outvar_pred_emb[:,start1:end1,start2:end2] + outvar_pred_emb_mode[:,index:index+1].unsqueeze(1)

            else: 
                start11 = self.mypara.val_relative[index][0][0]//self.mypara.patch_size[0]
                end11 = self.mypara.val_relative[index][0][1]//self.mypara.patch_size[0]
                start12 = self.mypara.val_relative[index][0][2]//self.mypara.patch_size[1]
                end12 = self.mypara.val_relative[index][0][3]//self.mypara.patch_size[1]
                start21 = self.mypara.val_relative[index][1][0]//self.mypara.patch_size[0]
                end21 = self.mypara.val_relative[index][1][1]//self.mypara.patch_size[0]
                start22 = self.mypara.val_relative[index][1][2]//self.mypara.patch_size[1]
                end22 = self.mypara.val_relative[index][1][3]//self.mypara.patch_size[1]

                if start11==end11:
                    end11 = end11+1
                
                if start12==end12:
                    end12 = end12+1

                if start21==end21:
                    end21 = end21+1

                if start22==end22:
                    end22 = end22+1

                enc_out[:,start11:end11,start12:end12] = enc_out[:,start11:end11,start12:end12] + enc_out_mode[:,index:index+1].unsqueeze(1)
                enc_out[:,start21:end21,start22:end22] = enc_out[:,start21:end21,start22:end22] - enc_out_mode[:,index:index+1].unsqueeze(1)
                outvar_pred_emb[:,start11:end11,start12:end12] = outvar_pred_emb[:,start11:end11,start12:end12] + outvar_pred_emb_mode[:,index:index+1].unsqueeze(1)
                outvar_pred_emb[:,start21:end21,start22:end22] = outvar_pred_emb[:,start21:end21,start22:end22] - outvar_pred_emb_mode[:,index:index+1].unsqueeze(1)

        enc_out_mode = enc_out.reshape(B, -1, self.mypara.his_len, enc_out.shape[-1])
        outvar_pred_emb_mode = outvar_pred_emb.reshape(B, -1, outvar_pred_emb.shape[-2], outvar_pred_emb.shape[-1])

        pred, enc_out, outvar_pred_emb = self.forward_sep(predictor, timestamps, self.encoder, self.decoder,self.linear_output, self.predictor_emb, self.predictand_emb, self.cube_dim, self.mypara.patch_size, in_mask, enout_mask, train, sv_ratio, bias_enc = enc_out_mode, bias_dec = outvar_pred_emb_mode)

        pred_mode = pred_mode.squeeze(-1).squeeze(2).permute(0,2,1)

        return pred, pred_mode


    def encode(self, encoder, predictor, timestamps, cube_dim, in_mask, predictor_emb, patch_size, bias_enc=None):
        """
        predictor: (B, lb, C, H, W)
        en_out: (Batch, S, lb, d_size)
        """

        lb = predictor.size(1)
        predictor = unfold_func(predictor, patch_size)
        
        predictor = predictor.reshape(predictor.size(0), lb, cube_dim, -1).permute(
            0, 3, 1, 2
        )
        predictor = predictor_emb(predictor, timestamps)
        if bias_enc is not None:
            assert bias_enc.shape == predictor.shape
            predictor += bias_enc

        en_out = encoder(predictor, in_mask)
        return en_out


    def decode(self, decoder, linear_output, predictand, timestamps, cube_dim, en_out, out_mask, enout_mask,predictand_emb, patch_size, bias_dec=None):
        """
        mypara:
            predictand: (B, pre_len, C, H, W)
        output:
            (B, pre_len, C, H, W)
        """
        H, W = predictand.size()[-2:]
        T = predictand.size(1)
        predictand = unfold_func(predictand, patch_size)
        predictand = predictand.reshape(
            predictand.size(0), T, cube_dim, -1
        ).permute(0, 3, 1, 2)
        predictand = predictand_emb(predictand, timestamps)
        # predictand torch.Size([128, 84, 1, 256])
        if bias_dec is not None:
            predictand += bias_dec[:,:,:predictand.shape[2]]
        output = decoder(predictand, en_out, out_mask, enout_mask)
        output_emb = output.clone()
        output = linear_output(output).permute(0, 2, 3, 1)
        output = output.reshape(
            predictand.size(0),
            T,
            cube_dim,
            H // patch_size[0],
            W // patch_size[1],
        )
        output = fold_func(
            output, output_size=(H, W), kernel_size=patch_size
        )
        return output, output_emb

    def make_mask_matrix(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 0).T
        return mask.to(self.mypara.device)


class multi_enc_layer(nn.Module):
    def __init__(self, enc_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([enc_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class multi_dec_layer(nn.Module):
    def __init__(self, dec_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([dec_layer for _ in range(num_layers)])

    def forward(self, x, en_out, out_mask, enout_mask):
        for layer in self.layers:
            x = layer(x, en_out, out_mask, enout_mask)
        return x
