from train_superres import main_superres


if __name__ == "__main__":

    # Choose dataset
    target_data = "mnist"

    # Choose hyperparameters
    diffusion_steps = 1000
    lr = 1E-4
    num_model_channels = 128
    num_res_blocks = 3
    batch_size = 8
    microbatch = 4
    use_fp16 = False
    dropout = 0.1
    learn_sigma = True
    loss_name = "rescaled_mse"
    schedule_sampler = "loss-second-moment"
    noise_schedule = "cosine"
    class_cond = False

    # Specify model folder URL
    url_folder_experiment = './models_data/first_try_superres/'

    # Check inputs
    if target_data not in {"mnist", "sar"}:
        raise ValueError("The specified target data is not supported.")

    # Specify data directories, image shape and number of data classes
    if target_data == "mnist":
        data_train_dir = './data/mnist_train/'
        data_val_dir = './data/mnist_test/'
        image_size = 32
        image_size_lr = 16
        num_input_channels = 1
        num_class = 10
        crop = False
    elif target_data == "sar":
        data_train_dir = './data/sar_tengeop/train/'
        data_val_dir = './data/sar_tengeop/test/'
        image_size = 256
        image_size_lr = 128
        num_input_channels = 1
        num_class = 10
        crop = True

    main_superres(
        data_train_dir=data_train_dir,
        data_val_dir=data_val_dir,
        schedule_sampler=schedule_sampler,
        lr=lr,
        batch_size=batch_size,
        microbatch=microbatch,
        use_fp16=use_fp16,
        image_size=image_size,
        image_size_lr=image_size_lr,
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
