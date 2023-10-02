import os

from train_lowres import main


if __name__ == "__main__":

    # Choose hyperparameters
    diffusion_steps = 2000
    lr = 1E-4
    num_model_channels = 128
    num_res_blocks = 4
    batch_size = 8
    microbatch = 4
    use_fp16 = False
    dropout = 0.1
    learn_sigma = True
    loss_name = "rescaled_mse"
    schedule_sampler = "loss-second-moment"
    noise_schedule = "cosine"
    class_cond = True

    # Specify model folder URL
    url_folder_experiment = './models_data/sar_lowres_try/'

    # Specify data directories, image shape and number of data classes
    if not os.path.exists('./data/sar_tengeop'):
        raise ValueError("The specified target data is not available. Please download the data first.")
    data_train_dir = './data/sar_tengeop/train/'
    data_val_dir = './data/sar_tengeop/val/'
    type_dataset = "image"
    image_size = 128
    image_size_hr = 256
    num_input_channels = 1
    num_class = 10
    crop = True

    main(
        data_train_dir=data_train_dir,
        data_val_dir=data_val_dir,
        schedule_sampler=schedule_sampler,
        lr=lr,
        batch_size=batch_size,
        microbatch=microbatch,
        use_fp16=use_fp16,
        image_size=image_size,
        image_size_hr=image_size_hr,
        num_input_channels=num_input_channels,
        num_model_channels=num_model_channels,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        learn_sigma=learn_sigma,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        url_folder_experiment=url_folder_experiment,
        class_cond=class_cond,
        num_class=num_class,
        loss_name=loss_name,
        crop=crop)
