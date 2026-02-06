"""
app_test.py

Model evaluation script for testing trained models on multiple datasets.
"""

import torch
from LoadData import *
from models import UniCM
from settings import setup_init, setup_testing_environment, load_pretrained_model, evaluate_on_dataset
from config import parse_args

torch.autograd.set_detect_anomaly(True)


def main(mypara):
    """
    Main evaluation function.
    
    Loads a trained model and evaluates it on multiple test datasets:
    - ERA5
    - ORAS5
    - SODA
    - GODAS
    """
    # Setup device
    device = torch.device(f"cuda:{mypara.cuda_id}" if torch.cuda.is_available() else "cpu")
    mypara.device = device
    
    # Setup paths
    file_path = f'{mypara.exp_folder}/SaveModel_Seed{mypara.seed}/'
    mypara.model_path = f'./experiments/{file_path}'
    
    # Parse patch size (needed for model initialization)
    mypara.patch_size = [int(i) for i in mypara.patch_size.split('-')]
    mypara.emb_spatial_size = 12 * 72 // (mypara.patch_size[0] * mypara.patch_size[1])

    """
    Setup testing environment and load model
    """
    setup_testing_environment(mypara)
    
    print(f'Model path: {mypara.model_path}')
    
    """
    Model initialization
    """
    mymodel = UniCM(mypara).to(device)
    
    model_size = sum(p.numel() * p.element_size() for p in mymodel.parameters()) 
    model_size_mb = model_size / (1024 ** 2)
    print(f"\nModel size: {model_size_mb:.2f} MB\n")
    
    # Load pretrained best model
    load_pretrained_model(mymodel, mypara, file_path)
    mymodel.mode = 'testing'
    
    # Write evaluation header
    with open(mypara.model_path+mypara.result_filename, 'w') as f:
        f.write('testing approach\n')
    
    """
    Evaluate on multiple datasets
    """
    evaluate_on_dataset('ERA5', mypara, mymodel, device, make_test_data_ERA5, 
                        use_auxiliary=True, auxiliary_slice=slice(-2, None))
    
    evaluate_on_dataset('ORAS5', mypara, mymodel, device, make_test_data_ORAS5)
    
    evaluate_on_dataset('SODA', mypara, mymodel, device, make_test_data_SODA224, 
                        use_auxiliary=True, load_hc=True)
    
    evaluate_on_dataset('GODAS', mypara, mymodel, device, make_test_data_GODAS,
                        use_auxiliary=True, load_hc=True, data_name='tos')
    
    print(f'Testing completed for {mypara.model_path}')


if __name__ == "__main__":
    mypara = parse_args()
    setup_init(mypara.seed)
    main(mypara)
