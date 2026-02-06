
for seed in $(seq 1 20)
do
    python app_test.py --batch_size 32 --cuda_id 0  --lambda3 1 --lambda2 0.01 --lambda1 1  --lr 5e-4  --dropout 0.2   --mode testing --training_data CESM2-FV2*gr  --seed $seed --exp_folder 'runs' --result_filename 'test_result_all.txt' --epochs 2
done

python app_ensemble.py --batch_size 32 --cuda_id 0 --climate_mode all --lambda3 1 --lambda2 0.01 --lambda1 1 --lr 5e-4 --dropout 0.2 --mode testing --training_data CESM2-FV2*gr --patch_size '2-2' --exp_folder 'runs' --num_ensemble 1 --result_filename 'test_result_ensemble.txt' --pretrained_path 'runs/SaveModel'
# note that the pretrained_path should be the path of the model that you want to use for ensemble
