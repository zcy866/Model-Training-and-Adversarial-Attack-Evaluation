**clone the environment:**

* create environment with python==3.9.0
* install torch according to torch==1.10.1+cu111
* install package with pip install -r requirements.txt

**prepare dataset:**

* place the data in the datasets directory, making sure to structure it as datasets->attack_dataset->clean_samples, adv_samples and train. Make sure there are no text files in these folders.

**training model:**

* script: python train_all.py <\name> --data_dir ./datasets --trial_seed <\seed>  --algorithm <\algorithm_name> --checkpoint_freq <\checkpoint_freq> --steps <\training_iterations> --img_model <\model_name> --lr <\learning_rate> --weight_decay <\weight_decay> --optimizer <\optimizer_name> --batch_size <\batch_size> --data <\data_split_type>
* It is recommended that <\name> and <\model_name> be consistent to facilitate model loading at test time.
* options of img_model: ResNet-50, mae-B, swin_transformer-B, swin_transformer-S, convnext-S, convnext-B, DINO_V2_VIT-S, DINO_V2_VIT-B, CLIP_img_encoder_VIT-L, CLIP_img_encoder_VIT-B
* options of algorithm_name: ERM,  Linear_Prob, LP_FT
* options of optimizer_name: adam, sgd, adamw
* options of data_split_type: train_train_val_clean, train_const_val_const
* example: python train_all.py ResNet_50 --data_dir ./datasets --trial_seed 0 --algorithm ERM --checkpoint_freq 1000 --steps 10000 --img_model ResNet-50 --lr 5e-5 --weight_decay 1e-4
* Important: the model will be saved in ./save_model

**compute attack score for single model and single adversarial samples:**

* script: python single_test.py <\name> --data_dir ./datasets --trial_seed <\seed>  --algorithm <\algorithm_name> --swad False --blur_scale <\blur_scale> --img_model <\model_name> --data <\data_split_type> --each_sample_score_record_folder <\the_folder_that_save_the_record_file_of_each_sample> --adv_sample_name </name_or_child_path_name_of_adversarial samples>
* example: python single_test.py convnext-S_clean  --data_dir ./datasets --trial_seed 0 --algorithm ERM --swad False --blur_scale 0.1 --img_model convnext-S --each_sample_score_record_folder ./score_record --adv_sample_name adv_samples

**select the best subset of adversarial samples from all adversarial samples based on multiple models:**

* script: python test_all.py <\name_list> --data_dir ./datasets --trial_seed <\seed> --all_algorithm <\algorithm_name_list> --swad False --blur_scale <\blur_scale> --all_img_model <\model_name_list> --each_sample_score_record_folder <\the_folder_that_save_the_record_file_of_each_sample> --goal_adv_name <\the_folder_name_that_save_selected_adv_samples> --adv_data_dir_name <\the_folder_name_that_save_all_adv_samples_to_be_selected> --training_mode
* example: python test_all.py ResNet-50_clean,convnext-S_clean --data_dir ./datasets --trial_seed 0 --all_algorithm ERM,ERM --swad False --blur_scale 0.1 --all_img_model ResNet-50,convnext-S --each_sample_score_record_folder ./score_record --goal_adv_name adv_samples --adv_data_dir_name all_adv_samples --training_mode
* important: with --training_mode means the sample selection mode is adopted and without it means validation mode is adopted
* TODO: add validation models

**analyze model:**
* script: python analysis_tool.py <\name> --data_dir ./datasets --trial_seed 0 --algorithm <\algorithm_name> --swad False --data <\data_split_type>
* example: python analysis_tool.py ResNet-50_val --data_dir ./datasets --trial_seed 0 --algorithm ERM --swad False --data train_const_val_const
