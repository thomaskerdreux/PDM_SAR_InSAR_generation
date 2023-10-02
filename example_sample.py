import os

from sample import main_sample


if __name__ == "__main__":

    # NOTE for RGB images, 'to_0_255' must be set to True
    url_model = "./models_data/SAR_lowres_128_cond_sigma_2000/ema_rate=0.99_78.pt"

    num_samples = 32
    sample_class = 5
    batch_size = 4
    use_ddim = False
    diffusion_steps = 500
    max_clip_val = 1
    plot = False

    url_save_path = './data/samples/SAR_superres_128_to_256_cond_sigma_2000/'

    # Check model path
    if not os.path.exists(url_model):
        raise ValueError('The "url_model" given as input does not exist.')

    if not os.path.exists(url_save_path):
        os.makedirs(url_save_path, exist_ok=True)

    main_sample(
        model_path=url_model,
        clip_denoised=True,
        max_clip_val=max_clip_val,
        num_samples=num_samples,
        sample_class=sample_class,
        batch_size=batch_size,
        use_ddim=use_ddim,
        diffusion_steps=diffusion_steps,
        url_save_path=url_save_path,
        to_0_255=False,
        plot=plot)
