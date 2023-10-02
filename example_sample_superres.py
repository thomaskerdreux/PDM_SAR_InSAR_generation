import os

from sample_superres import main_sample_superres


if __name__ == "__main__":
    '''
    For superresolution models, you need to first train a low-resolution model
    with (example_train_lowres.py) and then a superresolution model with
    (example_train_superres.py).

    Otherwise, load the pretrained model `SAR_lowres_128_cond_sigma_2000.zip` 
    and `SAR_superres_128_to_256_cond_sigma_2000.zip`
    as described in the ReadMe.
    
    Use (example_sample.py) to sample from the low-resolution model
    and save samples as `.npz` files, and then (example_sample_superres.py)
    to sample from the superresolution model, conditioned on the
    low-resolution samples.
    '''

    url_model = "./models_data/SAR_superres_128_to_256_cond_sigma_2000/ema_rate=0.99_33.pt"

    batch_size = 8
    use_ddim = False
    diffusion_steps = 500
    max_clip_val = 1
    plot = True
    url_save_path = './images/samples_SAR_cond_superres_500/'

    # Check model path
    if not os.path.exists(url_model):
        raise ValueError('The "url_model" given as input does not exist.')

    url_lr_data = './data/samples/SAR_superres_128_to_256_cond_sigma_2000/samples_32x128x128x1.npz'

    if not os.path.exists(url_lr_data):
        raise ValueError('First run the script example_sample.py).')

    main_sample_superres(
        model_path=url_model,
        url_lr_data=url_lr_data,
        clip_denoised=True,
        max_clip_val=max_clip_val,
        batch_size=batch_size,
        use_ddim=use_ddim,
        plot=plot,
        diffusion_steps=diffusion_steps,
        url_save_path=url_save_path)
