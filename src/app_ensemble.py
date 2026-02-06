"""
app_ensemble.py

Ensemble evaluation script for UniCM model.
Loads multiple trained models (Seeds 1-N) and computes ensemble metrics.
"""

import torch
import os
import numpy as np
from LoadData import *
from models import UniCM
from settings import setup_init, setup_testing_environment, validate_data_paths
from config import parse_args
from Trainer import TrainLoop

def evaluate_ensemble_on_dataset(dataset_name, mypara, model, device, load_func, **kwargs):
    print(f'Ensemble Evaluation on {dataset_name} dataset')
    
    # Ensure model path exists for saving results
    os.makedirs(mypara.model_path, exist_ok=True)

    # Load main test data
    data_name = kwargs.get('data_name', None)
    if data_name is not None:
        test_data = load_func(mypara).dataloader_seq(name=data_name)
    else:
        test_data = load_func(mypara).dataloader_seq()

    # Load auxiliary data if needed
    if kwargs.get('use_auxiliary', False):
        test_data_assist = make_test_data_ORAS5(mypara).dataloader_seq()
        
        if kwargs.get('load_hc', False):
            if dataset_name == 'SODA':
                test_data_hc = make_test_data_SODA224(mypara).dataloader_seq(name='hc')
                test_data = [
                    torch.cat((
                        test_data[0], 
                        test_data_assist[0][:test_data[0].shape[0], :, 1:3], 
                        test_data_hc[0], 
                        test_data_assist[0][:test_data[0].shape[0], :, -1:]
                    ), dim=2),
                    test_data[1], test_data[2]
                ]
            elif dataset_name == 'GODAS':
                test_data_hc = make_test_data_GODAS(mypara).dataloader_seq(name='hc')
                test_data = [
                    torch.cat((
                        test_data[0][:test_data_assist[0].shape[0]], 
                        test_data_assist[0][:, :, 1:3], 
                        test_data_hc[0][:test_data_assist[0].shape[0]], 
                        test_data_assist[0][:, :, -1:]
                    ), dim=2),
                    test_data[1][:test_data_assist[0].shape[0]], 
                    test_data[2][:test_data_assist[0].shape[0]], 
                ]
        else:
            aux_slice = kwargs.get('auxiliary_slice', slice(-2, None))
            test_data = [
                torch.cat((test_data[0], test_data_assist[0][:, :, aux_slice]), dim=2),
                test_data[1], test_data[2]
            ]
            
    # Set current test dataset name (used in Trainer for saving files)
    mypara.current_test_data = dataset_name

    # Create TensorDataset and DataLoader
    # Trainer expects val_data to be a DataLoader
    test_dataset = torch.utils.data.TensorDataset(*test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        num_workers=4, 
        batch_size=mypara.batch_size, 
        shuffle=False
    )
    
    # Initialize Trainer and run ensemble evaluation
    trainer = TrainLoop(
        args=mypara, 
        model=model, 
        val_data=test_loader, 
        device=device
    )
    
    # Calling Evaluation_ensemble
    # epoch=0 is passed as a dummy value
    trainer.Evaluation_ensemble(epoch=0)


def main(mypara):
    # Validate data paths before any heavy initialization
    validate_data_paths(mypara, ['ERA5', 'ORAS5', 'SODA224', 'GODAS'])

    # Setup device
    device = torch.device(f"cuda:{mypara.cuda_id}" if torch.cuda.is_available() else "cpu")
    mypara.device = device

    setup_testing_environment(mypara)
    
    # Setup model path for ensemble results
    # Saving to experiments/<exp_folder>/Ensemble/
    mypara.model_path = f'./experiments/{mypara.exp_folder}/Ensemble/'
    if not os.path.exists(mypara.model_path):
        os.makedirs(mypara.model_path)
        
    print(f'Results will be saved to: {mypara.model_path}')

    # Parse patch size
    mypara.patch_size = [int(i) for i in mypara.patch_size.split('-')]
    mypara.emb_spatial_size = 12 * 72 // (mypara.patch_size[0] * mypara.patch_size[1])

    # Model initialization
    mymodel = UniCM(mypara).to(device)
    mymodel.mode = 'testing' # Set model mode
    
    # Evaluate on datasets
    
    # ERA5
    evaluate_ensemble_on_dataset('ERA5', mypara, mymodel, device, make_test_data_ERA5, 
                        use_auxiliary=True, auxiliary_slice=slice(-2, None))
    
    # ORAS5
    evaluate_ensemble_on_dataset('ORAS5', mypara, mymodel, device, make_test_data_ORAS5)
    
    # SODA
    evaluate_ensemble_on_dataset('SODA', mypara, mymodel, device, make_test_data_SODA224, 
                        use_auxiliary=True, load_hc=True)
    
    # GODAS
    evaluate_ensemble_on_dataset('GODAS', mypara, mymodel, device, make_test_data_GODAS,
                        use_auxiliary=True, load_hc=True, data_name='tos')

    print('Ensemble testing completed.')


if __name__ == "__main__":
    mypara = parse_args()
    setup_init(mypara.seed)
    main(mypara)
