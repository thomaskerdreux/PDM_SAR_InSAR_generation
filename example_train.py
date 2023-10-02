from train import main


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
    learn_sigma = False
    loss_name = "rescaled_mse"
    schedule_sampler = "loss-second-moment"
    noise_schedule = "cosine"

    # Specify model folder URL
    url_folder_experiment = './models_data/cifar10_try/'

    # Check inputs
    if target_data not in {"mnist", "cifar10", "sar", "insar_interfero",
                           "insar_noise"}:
        raise ValueError("The specified target data is not supported.")

    # Determine type_dataset
    if target_data in {"mnist", "cifar10", "sar"}:
        type_dataset = "image"
    elif target_data in {"insar_interfero", "insar_noise"}:
        type_dataset = "hdf5"

    # Specify data directories, image shape and number of data classes
    if target_data == "mnist":
        data_train_dir = './data/mnist_train/'
        data_val_dir = './data/mnist_val/'
        image_size = 32
        num_input_channels = 1
        num_class = 10
        crop = False
        class_cond = True
        list_keys_hdf5_original = None
        key_data = None
        key_other = None
    elif target_data == "cifar10":
        data_train_dir = './data/cifar_train/'
        data_val_dir = './data/cifar_val/'
        image_size = 32
        num_input_channels = 3
        num_class = 10
        crop = False
        class_cond = True
        list_keys_hdf5_original = None
        key_data = None
        key_other = None
    elif target_data == "sar":
        data_train_dir = './data/sar_tengeop/train/'
        data_val_dir = './data/sar_tengeop/val/'
        image_size = 128
        num_input_channels = 1
        num_class = 10
        crop = True
        class_cond = True
        list_keys_hdf5_original = None
        key_data = None
        key_other = None
    elif target_data == "insar_interfero":
        data_train_dir = './data/insar_unwrap/train/'
        data_val_dir = './data/insar_unwrap/val/'
        image_size = 128
        num_input_channels = 1
        crop = False
        class_cond = False
        num_class = None
        list_keys_hdf5_original = ['unwrap', 'dates']
        key_data = 'unwrap'
        key_other = 'dates'
    elif target_data == "insar_noise":
        data_train_dir = './data/insar_noise/train/'
        data_val_dir = './data/insar_noise/val/'
        image_size = 32
        num_input_channels = 1
        crop = False
        class_cond = False
        num_class = None
        list_keys_hdf5_original = ['real_data', 'count']
        key_data = 'real_data'
        key_other = 'count'

    if class_cond and target_data in {"insar_interfero", "insar_noise"}:
        raise ValueError(f"{target_data} is incompatible with class_cond = True.")

    main(
        type_dataset=type_dataset,
        data_train_dir=data_train_dir,
        data_val_dir=data_val_dir,
        schedule_sampler=schedule_sampler,
        lr=lr,
        batch_size=batch_size,
        microbatch=microbatch,
        use_fp16=use_fp16,
        image_size=image_size,
        num_input_channels=num_input_channels,
        num_model_channels=num_model_channels,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        learn_sigma=learn_sigma,
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        url_folder_experiment=url_folder_experiment,
        list_keys_hdf5_original=list_keys_hdf5_original,
        key_data=key_data,
        key_other=key_other,
        class_cond=class_cond,
        num_class=num_class,
        loss_name=loss_name,
        crop=crop)
