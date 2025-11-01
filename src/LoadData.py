"""
LoadData.py

This module handles loading and preprocessing of climate datasets (CMIP6, ORAS5, GODAS, SODA).

Key Classes:
- make_data: Base class for loading and preprocessing CMIP6 data.
- make_data_ORAS5: Subclass for loading and preprocessing ORAS5 reanalysis data.
- make_data_GODAS: Subclass for loading and preprocessing GODAS reanalysis data.
- make_test_data_SODA: Subclass for loading and preprocessing SODA reanalysis data.

The classes handle:
- Reading NetCDF files using xarray.
- Spatiotemporal cropping and coarsening (downsampling).
- Handling NaNs and outliers.
- Generating sequences (history + prediction) for the model.
"""

import numpy as np
import xarray as xr
import random
import os
import torch


class make_data():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (60//reso, 120//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/CMIP6-Multivariate"

    def data_process(self, filename, name):

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)[name]

        if name=='t20d':
            da = da[:,:180,:360]

        if name in ['tauu','tauv']: 
            da_interp = da.coarsen(lat=self.reso, lon=self.reso//2).mean()

        else:
            da_interp = da.coarsen(lat=self.reso, lon=self.reso).mean()

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            ...,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = np.expand_dims(da_interp, axis=0)

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        return data, std

    def dataloader_seq(self, filename, name='tos'):
        history = len(self.mypara.template)
        data, std = self.data_process(filename, name)
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1
        
        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]


class make_val_data_SODA224():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/SODA224"

    def data_process(self, filename='soda_224_1876_2011.nc'):

        reso  = self.reso

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)["temp"][:,0] # 5m

        da_interp = da[..., :320, :720]
        da_interp = da_interp.coarsen(latitude=self.reso * 2, longitude=self.reso * 2).mean().values

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :(1979-1876)*12,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = np.expand_dims(da_interp, axis=0)

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self):
        history = len(self.mypara.template)
        data, std = self.data_process()
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]


class make_test_data_ERA5():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (60//reso, 120//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/"

    def data_process(self, filename='ERA5.nc',name='tos'):

        reso  = self.reso

        ds = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4')
        ds = ds.sortby("latitude").sel(valid_time=slice("1980-01-01", "2014-12-31"))

        ds = xr.concat([ds[var] for var in ["sst","u10","v10"]], dim='variable')

        ds = ds[:,:,:720,:1440]

        da_interp = ds.coarsen(latitude=self.reso * 4, longitude=self.reso * 4).mean().values

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :,
            :,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (100//reso-self.lat_range[0], 115//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (70//reso-self.lat_range[0], 90//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 50//reso, 70//reso),(80//reso-self.lat_range[0], 90//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((65//reso-self.lat_range[0], 80//reso-self.lat_range[0], 65//reso, 80//reso),(60//reso-self.lat_range[0], 80//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (95//reso-self.lat_range[0], 115//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (80//reso-self.lat_range[0], 90//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self, name='tos'):
        history = len(self.mypara.template)
        data, std = self.data_process(name=name)
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]


class make_train_data_ERA5():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (60//reso, 120//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/"

    def data_process(self, filename='ERA5.nc',name='tos'):

        reso  = self.reso

        ds = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4')

        ds = ds.sortby("latitude").sel(valid_time=slice("1958-01-01", "1979-12-31"))

        ds = xr.concat([ds[var] for var in ["sst","u10","v10"]], dim='variable')

        ds = ds[:,:,:720,:1440]

        da_interp = ds.coarsen(latitude=self.reso * 4, longitude=self.reso * 4).mean().values

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :,
            :,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (100//reso-self.lat_range[0], 115//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (70//reso-self.lat_range[0], 90//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 50//reso, 70//reso),(80//reso-self.lat_range[0], 90//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((65//reso-self.lat_range[0], 80//reso-self.lat_range[0], 65//reso, 80//reso),(60//reso-self.lat_range[0], 80//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (95//reso-self.lat_range[0], 115//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (80//reso-self.lat_range[0], 90//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (85//reso-self.lat_range[0], 95//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self, name='tos'):
        history = len(self.mypara.template)
        data, std = self.data_process(name=name)
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]


class make_test_data_SODA224():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/SODA224"

    def data_process(self, filename='soda_224_1876_2011.nc',name='tos'):

        reso  = self.reso

        if name == 'tos':
            da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)["temp"][:,0] # 5m
        elif name == 'hc':
            da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)["temp"]
            da = da.mean(dim='LEV')

        da_interp = da[..., :320, :720]
        da_interp = da_interp.coarsen(latitude=self.reso * 2, longitude=self.reso * 2).mean().values

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            (1980-1876)*12:,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = np.expand_dims(da_interp, axis=0)

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self, name='tos'):
        history = len(self.mypara.template)
        data, std = self.data_process(name=name)
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]



class make_test_data_GODAS():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/GODAS-new"

    def data_process(self, filename='pottemp_1980_2021.nc',name='tos'):

        if name == 'tos':
            da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)["pottmp"][:,0]

        elif name == 'hc':
            da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)["pottmp"][:,:26]
            da = da.mean(dim='level')

        reso  = self.reso

        lat_new = np.arange(-74.5, 60.5, 1)
        lon_new = np.arange(0.5, 360.5, 1)

        da_interp = da.interp(lat=lat_new, lon=lon_new)
        da_interp = da_interp.coarsen(lat=self.reso, lon=self.reso).mean().values

        da_interp = np.nan_to_num(da_interp)
        da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            ...,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = np.expand_dims(da_interp, axis=0)

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self, name = 'tos'):
        history = len(self.mypara.template)
        data, std = self.data_process(name=name)
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]




class make_test_data_ORAS5():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/ORAS5"

    def data_process(self, filename='ORAS5_1958_2014.nc'):

        reso  = self.reso

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)

        vars_to_merge = ['sosstsst', 'sozotaux', 'sometauy', 'sohtc300']

        if self.mypara.input_channal==5 or self.mypara.t20d_mode==1:
            vars_to_merge.append('so20chgt')

        da = xr.concat([da[var] for var in vars_to_merge], dim='variable')

        da_interp = da[:, :, :320, :720]
        da_interp = da_interp.coarsen(lat=self.reso, lon=self.reso).mean().values

        da_interp = np.nan_to_num(da_interp)
        #da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :,
            (1980-1958)*12:,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)

        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]

        return data, std

    def dataloader_seq(self):
        history = len(self.mypara.template)
        data, std = self.data_process()
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]

class make_test_data_ORAS5_2015_2025():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/ORAS5"

    def data_process(self, filename='ORAS5_1958_2025.nc'):

        reso  = self.reso

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)

        vars_to_merge = ['sosstsst', 'sozotaux', 'sometauy', 'sohtc300']

        if self.mypara.input_channal==5 or self.mypara.t20d_mode==1:
            vars_to_merge.append('so20chgt')

        da = xr.concat([da[var] for var in vars_to_merge], dim='variable')

        da_interp = da[:, :, :320, :720]
        da_interp = da_interp.coarsen(lat=self.reso, lon=self.reso).mean().values

        da_interp = np.nan_to_num(da_interp)
        #da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :,
            (2015-1958)*12:(2025-1958)*12:,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)

        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]

        return data, std

    def dataloader_seq(self):
        history = len(self.mypara.template)
        data, std = self.data_process()
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]

class make_train_data_ORAS5():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/ORAS5"

    def data_process(self, filename='ORAS5_1958_2014.nc'):

        reso  = self.reso

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)

        vars_to_merge = ['sosstsst', 'sozotaux', 'sometauy', 'sohtc300']

        if self.mypara.input_channal==5 or self.mypara.t20d_mode==1:
            vars_to_merge.append('so20chgt')

        da = xr.concat([da[var] for var in vars_to_merge], dim='variable')

        da_interp = da[:, :, :320, :720]
        da_interp = da_interp.coarsen(lat=self.reso, lon=self.reso).mean().values

        da_interp = np.nan_to_num(da_interp)
        #da_interp[abs(da_interp) > 999] = 0

        da_interp = da_interp[
            :,
            :(1980-1958)*12,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])

        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        print(filename, data[0].shape)

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)

        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]

        return data, std

    def dataloader_seq(self):
        history = len(self.mypara.template)
        data, std = self.data_process()
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]

class make_val_data_ORAS5():
    def __init__(self, mypara):
        self.mypara = mypara

        self.reso = mypara.resolution

        reso = self.reso

        self.lat_range = (45//reso, 105//reso)
        self.lon_range = (0//reso, 360//reso)

        self.folder_path = "../dataset/ORAS5"

    def data_process(self, filename='ORAS5_1958_2014.nc'):

        reso  = self.reso

        da = xr.open_dataset(self.folder_path + '/' + filename, engine = 'netcdf4',decode_times=False)

        vars_to_merge = ['sosstsst', 'sozotaux', 'sometauy', 'sohtc300']

        if self.mypara.input_channal==5 or self.mypara.t20d_mode==1:
            vars_to_merge.append('so20chgt')

        da = xr.concat([da[var] for var in vars_to_merge], dim='variable')

        da_interp = da[:, :, :320, :720]
        da_interp = da_interp.coarsen(lat=self.reso, lon=self.reso).mean().values

        da_interp = np.nan_to_num(da_interp)

        da_interp = da_interp[
            :(1980-1958)*12,
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ]

        da_interp = da_interp.reshape(da_interp.shape[0], -1, 12, da_interp.shape[-2], da_interp.shape[-1])

        mean = np.mean(da_interp,axis=1, keepdims=True)
        da_interp = (da_interp - mean).reshape(da_interp.shape[0], -1, da_interp.shape[-2], da_interp.shape[-1])


        mask = np.all(da_interp == 0, axis=1, keepdims=True)
        valid_data = np.where(~mask, da_interp, np.nan)
        std = np.nanstd(valid_data,axis=(1,2,3),keepdims=True)
        std = torch.tensor(std)

        if self.mypara.norm_std==1:
            da_interp = da_interp / std
            std = std[0,0,0,0]
        else:
            std = torch.ones_like(std)

        da_interp = np.transpose(da_interp, (1,0,2,3))

        data = (torch.tensor(da_interp), torch.tensor([i % 12 for i in range(da_interp.shape[0])]).int())

        self.mypara.val_nino_relative = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 190//reso, 240//reso)
        self.mypara.val_NPMM_relative = (85//reso-self.lat_range[0], 100//reso-self.lat_range[0], 200//reso, 240//reso)
        self.mypara.val_SPMM_relative = (50//reso-self.lat_range[0], 60//reso-self.lat_range[0], 250//reso, 270//reso)
        self.mypara.val_IOB_relative = (55//reso-self.lat_range[0], 75//reso-self.lat_range[0], 40//reso, 100//reso)
        self.mypara.val_IOD_relative = ((65//reso-self.lat_range[0], 85//reso-self.lat_range[0], 50//reso, 70//reso),(65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 90//reso, 110//reso))
        self.mypara.val_SIOD_relative = ((50//reso-self.lat_range[0], 65//reso-self.lat_range[0], 65//reso, 80//reso),(45//reso-self.lat_range[0], 65//reso-self.lat_range[0], 90//reso, 120//reso))
        self.mypara.val_TNA_relative = (80//reso-self.lat_range[0], 100//reso-self.lat_range[0], 305//reso, 345//reso)
        self.mypara.val_nino12 = (65//reso-self.lat_range[0], 75//reso-self.lat_range[0], 270//reso, 280//reso)
        self.mypara.val_nino3 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 210//reso, 270//reso)
        self.mypara.val_nino4 = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 200//reso, 210//reso)
        self.mypara.WWV = (70//reso-self.lat_range[0], 80//reso-self.lat_range[0], 120//reso, 280//reso)

        if self.mypara.mode_interaction == '0':
            self.mypara.val_relative = [self.mypara.val_nino_relative]
        elif self.mypara.mode_interaction == '1':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_NPMM_relative, self.mypara.val_SPMM_relative, self.mypara.val_IOB_relative, self.mypara.val_IOD_relative, self.mypara.val_SIOD_relative, self.mypara.val_TNA_relative, self.mypara.val_nino12, self.mypara.val_nino3, self.mypara.val_nino4]
        elif self.mypara.mode_interaction == '2':
            self.mypara.val_relative = [self.mypara.val_nino_relative, self.mypara.val_IOD_relative]
        
        return data, std

    def dataloader_seq(self):
        history = len(self.mypara.template)
        data, std = self.data_process()
        train_data, train_ts, his_all_data, his_all_ts, std_all, index_all = [], [], [], [], [], []
        data, ts = data
        num_samples = data.shape[0] - (self.mypara.his_len + self.mypara.pred_len_all) + 1

        for t in range(0, num_samples):
            index = (t, t+self.mypara.his_len, t+self.mypara.his_len+self.mypara.pred_len_all)
            train_data.append(data[index[0]:index[2]])
            train_ts.append(ts[index[0]:index[2]])
            std_all.append(std)
            if self.mypara.test_template==1:
                self.mypara.template.append(data[index[0]:index[2]])
            index_all.append(t)
        train_data = torch.stack(train_data)
        train_ts = torch.stack(train_ts)
        std_all = torch.stack(std_all)
        index_all = torch.tensor(index_all)
        return [train_data, train_ts, std_all, index_all]