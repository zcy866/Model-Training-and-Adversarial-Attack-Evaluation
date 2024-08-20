**clone the environment:**

* create environment with python==3.9.0
* install torch according to torch==1.10.1+cu111
* install package with pip install -r requirements.txt

**prepare dataset:**

* place the data in the datasets directory, making sure to structure it as datasets->attack_dataset->clean_samples, adv_samples and train. Make sure there are no text files in these folders.

**training model:**

* script: python train_all.py <\name> --data_dir ./datasets --trial_seed 0 --algorithm <\algorithm_name> --checkpoint_freq <\checkpoint_freq> --steps <\training_iterations> --img_model <\model_name> --lr <\learning_rate> --weight_decay <\weight_decay> --optimizer <\optimizer_name> --batch_size <\batch_size> --data <\data_split_type>
* It is recommended that <\name> and <\model_name> be consistent to facilitate model loading at test time.
* options of img_model: ResNet-50, mae-B, swin_transformer-B, swin_transformer-S, convnext-S, convnext-B, DINO_V2_VIT-S, DINO_V2_VIT-B, CLIP_img_encoder_VIT-L, CLIP_img_encoder_VIT-B
* options of algorithm_name: ERM,  Linear_Prob, LP_FT
* options of optimizer_name: adam, sgd, adamw
* options of data_split_type: train_train_val_clean, train_const_val_const
* example: python train_all.py ResNet_50 --data_dir ./datasets --trial_seed 0 --algorithm Linear_Prob --checkpoint_freq 1000 --steps 10000 --img_model ResNet-50 --lr 5e-5 --weight_decay 1e-4
* Important: the model will be saved in ./save_model

**compute attack score:**

* script: python test_all.py <\name> --data_dir ./datasets --trial_seed 0 --algorithm <\algorithm_name> --swad False --blur_scale <\blur_scale> --data <\data_split_type>
* example: python test_all.py ResNet_50 --data_dir ./datasets --trial_seed 0 --algorithm Linear_Prob --swad False --blur_scale 0.1
model ResNet-50 --lr 5e-5 --weight_decay 1e-4

**analyze model:**
* script: python test_all.py <\name> --data_dir ./datasets --trial_seed 0 --algorithm <\algorithm_name> --swad False --data <\data_split_type>