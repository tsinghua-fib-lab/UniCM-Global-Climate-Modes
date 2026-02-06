"""
settings.py

This module contains all setup and initialization functions for training environment.

Functions:
- setup_init: Initialize random seeds and PyTorch settings for reproducibility
- setup_training_environment: Setup device, paths, directories, and parameters
- setup_testing_environment: Setup testing environment parameters
- load_pretrained_model: Load pretrained model from checkpoint
- evaluate_on_dataset: Evaluate model on a specific dataset
"""


# ... (imports)
import random
import os
import numpy as np
import torch
from my_tools import evaluation_test_data
from LoadData import make_val_data_ORAS5


def setup_init(seed):
    """
    Initialize random seeds and configure PyTorch for reproducibility.
    
    Args:
        seed: Random seed value for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_training_environment(mypara):
    """
    Setup training environment including device, paths, directories, and parameter initialization.
    
    Args:
        mypara: Parameter object containing training configuration
        
    Returns:
        device: PyTorch device (CUDA or CPU)
    """
    
    # Setup device
    device = torch.device(f"cuda:{mypara.cuda_id}" if torch.cuda.is_available() else "cpu")
    mypara.device = device
    
    # Setup paths
    file_path = f'{mypara.exp_folder}/SaveModel_Seed{mypara.seed}/'
    mypara.model_path = f'./experiments/{file_path}'
    log_path = f'./logs/{file_path}'
    
    print(f'File path: {file_path}, Input channels: {mypara.input_channal}')
    
    # Create necessary directories
    os.makedirs('./experiments/', exist_ok=True)
    os.makedirs(mypara.model_path, exist_ok=True)
    os.makedirs(f'{mypara.model_path}model_save/', exist_ok=True)
    
    # Initialize parameter storage
    mypara.train_scaler = {}
    mypara.val_scaler = {}
    mypara.template = []
    
    # Parse patch size and compute embedding spatial size
    mypara.patch_size = [int(i) for i in mypara.patch_size.split('-')]
    mypara.emb_spatial_size = 12 * 72 // (mypara.patch_size[0] * mypara.patch_size[1])
    
    return device


def setup_testing_environment(mypara):
    """
    Setup testing environment by adjusting parameters for evaluation mode.
    
    Args:
        mypara: Parameter object
    """
    mypara.mode = 'testing'
    mypara.batch_size = mypara.batch_size // 2

    make_val_data_ORAS5(mypara).dataloader_seq()
    


def load_pretrained_model(model, mypara, file_path=''):
    """
    Load pretrained model from checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        mypara: Parameter object containing model path configuration
        file_path: Optional custom file path for model checkpoint
        
    Returns:
        bool: True if model was loaded successfully, False otherwise
    """
    # Determine pretrained model path
    if mypara.pretrained_path == '':
        pretrained_path = f'./experiments/{file_path}model_save/model_best.pkl'
    else:
        pretrained_path = f'./experiments/{mypara.pretrained_path}/model_save/model_best.pkl'
    
    # Load model if checkpoint exists
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        print(f'Loaded finetuned model from: {pretrained_path}')
        return True
    else:
        print('No pretrained model found, using model from scratch')
        return False


def evaluate_on_dataset(dataset_name, mypara, model, device, load_func, **kwargs):
    """
    Evaluate model on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ERA5', 'ORAS5', 'SODA', 'GODAS')
        mypara: Parameter object
        model: Model to evaluate
        device: PyTorch device
        load_func: Function to load the test data
        **kwargs: Additional arguments for data loading and processing
            - use_auxiliary: Whether to use auxiliary ORAS5 data
            - auxiliary_slice: How to slice auxiliary data
            - load_hc: Whether to load heat content data
            - data_name: Name parameter for load_func (e.g., 'tos', 'hc')
    """
    # Import here to avoid circular dependency
    from LoadData import make_test_data_ORAS5, make_test_data_SODA224, make_test_data_GODAS
    
    # Log to console and file
    print(f'Testing on {dataset_name} dataset')
    with open(f'{mypara.model_path}{mypara.result_filename}', 'a') as f:
        f.write(f'\n{"-" * 32}dataset:{dataset_name}{"-" * 32}\n')
    
    
    # Load main test data
    data_name = kwargs.get('data_name', None)
    if data_name is not None:
        test_data = load_func(mypara).dataloader_seq(name=data_name)
    else:
        test_data = load_func(mypara).dataloader_seq()

    
    # Load auxiliary data if needed
    if kwargs.get('use_auxiliary', False):
        test_data_assist = make_test_data_ORAS5(mypara).dataloader_seq()
        
        # Handle heat content data if needed
        if kwargs.get('load_hc', False):
            if dataset_name == 'SODA':
                test_data_hc = make_test_data_SODA224(mypara).dataloader_seq(name='hc')
                print(test_data[0].shape, test_data_assist[0].shape, test_data_hc[0].shape)
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
            # ERA5 case
            aux_slice = kwargs.get('auxiliary_slice', slice(-2, None))
            test_data = [
                torch.cat((test_data[0], test_data_assist[0][:, :, aux_slice]), dim=2),
                test_data[1], test_data[2]
            ]
    
    # Print validation relative info and evaluate
    print(model.mypara.val_relative)
    mypara.current_test_data = dataset_name
    evaluation_test_data(test_data, mypara, model, device)



def evaluate_on_dataset_ensemble(dataset_name, mypara, model, device, load_func, **kwargs):
    """
    Evaluate multiple models (bagging/ensemble) on a specific dataset.
    
    This function loads multiple trained models and averages their predictions
    for more robust evaluation results.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ERA5', 'ORAS5', 'SODA', 'GODAS')
        mypara: Parameter object (must have num_ensemble and pretrained_path set)
        model: Model architecture (weights will be loaded from checkpoints)
        device: PyTorch device
        load_func: Function to load the test data
        **kwargs: Additional arguments for data loading (same as evaluate_on_dataset)
    """
    from LoadData import make_test_data_ORAS5, make_test_data_SODA224, make_test_data_GODAS
    from Trainer import TrainLoop
    
    # Log to console and file
    print(f'\n{"="*80}')
    print(f'Ensemble Evaluation on {dataset_name} dataset')
    print(f'Number of models: {mypara.num_ensemble}')
    print(f'{"="*80}\n')
    
    with open(f'{mypara.model_path}result_all_ensemble.txt', 'a') as f:
        f.write(f'\n{"-" * 32}dataset:{dataset_name}{"-" * 32}\n')
        f.write(f'Ensemble models: {mypara.num_ensemble}\n')
    
    # Load main test data
    data_name = kwargs.get('data_name', None)
    if data_name is not None:
        test_data = load_func(mypara).dataloader_seq(name=data_name)
    else:
        test_data = load_func(mypara).dataloader_seq()
    
    # Load auxiliary data if needed (same logic as evaluate_on_dataset)
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
                    test_data[2][:test_data_assist[0].shape[0]]
                ]
        else:
            aux_slice = kwargs.get('auxiliary_slice', slice(-2, None))
            test_data = [
                torch.cat((test_data[0], test_data_assist[0][:, :, aux_slice]), dim=2),
                test_data[1], test_data[2]
            ]
    
    # Set current test dataset
    mypara.current_test_data = dataset_name
    
    # Create trainer instance for bagging evaluation
    test_dataloader = torch.utils.data.TensorDataset(*test_data)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataloader, 
        num_workers=4, 
        batch_size=mypara.batch_size * 4, 
        shuffle=False
    )
    
    trainer = TrainLoop(
        args=mypara,
        model=model,
        val_data=test_dataloader,
        device=device,
    )
    
    # Run ensemble evaluation (uses Evaluation_ensemble from Trainer)
    print(f'Running ensemble evaluation with {mypara.num_ensemble} models...')
    trainer.Evaluation_ensemble(epoch=0)



