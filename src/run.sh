
# training
for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    python app.py --batch_size 50 --cuda_id 2 --machine LM2  --climate_mode all --mode_coef 0.01 --ours_coef 1 --vdt_coef 1  --lr 5e-5  --dropout 0.2   --mode training --training_data CESM2-FV2*gr --patch_size '2-2'  --mode_interaction 1  --input_channal 5 --norm_std 1 --t20d_mode 1 --seed $seed 
done


# evaluation (bagging)
python app.py --batch_size 50 --cuda_id 2 --machine LM2  --climate_mode all --mode_coef 0.01 --ours_coef 1 --vdt_coef 1  --lr 5e-5  --dropout 0.2   --mode testing --training_data CESM2-FV2*gr --patch_size '2-2'  --mode_interaction 1  --input_channal 5 --norm_std 1 --t20d_mode 1 --num_bagging 20  --pretrained_path SaveModel