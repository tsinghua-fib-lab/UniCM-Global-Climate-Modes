

# UniCM

This repository provides the official implementation of **UniCM**, a unified deep model for global climate modes forecasting, as presented in the paper, "**Learning the coupled dynamics of global climate modes**".

Global weather extremes, from monsoons to droughts, are shaped by a network of recurrent, coupled ocean-atmosphere patterns known as climate modes (e.g., El Niño-Southern Oscillation (ENSO), Indian Ocean Dipole (IOD)). Forecasting this interconnected global system—rather than treating modes in isolation—remains a fundamental challenge.

UniCM's key innovation is a **coupling-aware approach** that learns the dynamics of the entire coupled global system directly from data. It establishes a new state of the art, significantly outperforming previous models and extending the skillful forecast lead time for critical climate patterns.

## Key Features

  * **Unified Global Forecasting:** A single model that learns the coupled dynamics of global climate modes, capturing the non-linear interactions between patterns in the Pacific, Indian, and Atlantic oceans.
  * **Coupling-Aware Architecture:** Two spatio-temporal Transformers (Encoder-Decoder) designed to process multivariate climate data and model dependencies across different ocean basins.
  * **State-of-the-Art Performance:** Outperforms previous leading models in forecasting key climate modes, particularly for long lead times.

  
## Project Structure

```
.
├── app.py                  # Main entry point: argument parsing, train/test logic
├── run.sh                  # Example script for training and evaluation
├── models.py               # Core model definition (UniCM)
├── Trainer.py              # Contains the TrainLoop class for training and validation
├── LoadData.py             # Data loading and preprocessing classes
├── my_tools.py             # Utility functions (miniEncoder, miniDecoder, attention)
├── Embed.py                # Spatio-temporal token embedding classes
├── func_for_prediction.py  # Helper functions for final prediction and evaluation
└── ...
```

## Installation

### Environment

- Tested OS: Linux (Ubuntu 22.04.3 LTS, training and testing)
- Python: 3.11.9
- Pytorch: 2.0.1
- CUDA: 12.2

### Package Dependencies

This project depends on several Python libraries. You can install them using pip.

```
pip install -r requirements.txt
```

## Data Preparation

The data loaders in `LoadData.py` are configured to read preprocessed NetCDF (`.nc`) files.

By default, the `make_data` class expects data to be in a directory relative to the project, such as `../dataset/CMIP6-Multivariate`.

Please **modify the paths within `LoadData.py`** (e.g., `self.folder_path`) to match your dataset's location.

## Usage

The `run.sh` script provides a clear example of how to train and evaluate the model using an ensemble/bagging approach.

### 1\. Training (with Multiple Seeds)

The script first trains 20 separate models, each with a different random seed. This is the first step required for ensemble bagging.

```bash
# training
for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    python app.py --batch_size 50 --cuda_id 2 --machine LM2 \
                  --climate_mode all --mode_coef 0.01 --ours_coef 1 --vdt_coef 1 \
                  --lr 5e-5  --dropout 0.2 \
                  --mode training --training_data CESM2-FV2*gr \
                  --patch_size '2-2'  --mode_interaction 1 \
                  --input_channal 5 --norm_std 1 --t20d_mode 1 --seed $seed 
done
```

**Key Parameters:**

  * `--mode training`: Sets the script to training mode.
  * `--cuda_id 2`: Assigns the task to GPU 2.
  * `--seed $seed`: Passes the current loop's seed to the script.
  * `--input_channal 5`: Specifies 5 input variables (e.g., SST, T300, U, V, H).

### 2\. Evaluation (Ensemble Bagging)

After all 20 models are trained and saved, this command runs the model in `testing` mode. It loads all 20 models and computes the ensemble average for a final, robust evaluation.

```bash
# evaluation (bagging)
python app.py --batch_size 50 --cuda_id 2 --machine LM2 \
              --climate_mode all --mode_coef 0.01 --ours_coef 1 --vdt_coef 1 \
              --lr 5e-5  --dropout 0.2 \
              --mode testing --training_data CESM2-FV2*gr \
              --patch_size '2-2'  --mode_interaction 1 \
              --input_channal 5 --norm_std 1 --t20d_mode 1 \
              --num_bagging 20  --pretrained_path SaveModel
```

**Key Parameters:**

  * `--mode testing`: Sets the script to evaluation mode.
  * `--num_bagging 20`: Instructs the script to load 20 models for the ensemble.
  * `--pretrained_path SaveModel`: Specifies the directory where the 20 trained models are stored.

## Key Parameters (`app.py`)

  * `--mode`: `training` or `testing`. Sets the operational mode.
  * `--cuda_id`: Specifies the GPU ID to use.
  * `--his_len`: Length of the historical input sequence (in months).
  * `--pred_len`: Length of the prediction sequence (in months).
  * `--batch_size`: Batch size for training and evaluation.
  * `--lr`: Learning rate.
  * `--training_data`: Identifier for the training dataset.
  * `--patch_size`: Spatio-temporal patch size (e.g., `'2-2'`).
  * `--input_channal`: Number of input data channels/variables.
  * `--seed`: Random seed for reproducibility.
  * `--num_bagging`: (Testing only) Number of ensemble members for bagging.
  * `--pretrained_path`: (Testing only) Directory where pretrained models are saved.
  * `--mode_coef`, `--ours_coef`, `--vdt_coef`: Weights for the different components of the composite loss function.