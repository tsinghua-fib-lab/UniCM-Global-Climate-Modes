for seed in $(seq 1 20)
do
    python app_train.py --batch_size 32 --cuda_id 0  --lambda3 1 --lambda2 0.01 --lambda1 1  --lr 5e-4  --dropout 0.2   --mode training --training_data CESM2-FV2*gr  --seed $seed --exp_folder 'runs' --result_filename 'train_result_all.txt' --epochs 2
done