import os

from sample_DPM import (
    create_dict_input_ODE_method,
    compare_approaches_MNIST,
    sample_from_ODE
)


if __name__ == '__main__':

    type_dataset = 'MNIST'

    if type_dataset == 'MNIST':
        url_model_uncond = "./models_data/mnist_32_no_cond_sigma_100/ema_0.9999_23_vb=0.9521_mse=0.0013.pt"
        url_model_cond = "./models_data/mnist_32_cond_sigma_100/ema_0.9999_35_vb=0.9274_mse=0.0013.pt"

        # Because data in [-1, 1]
        max_clip_val = 1

    test_sample_DPM = False
    test_time_comparison = True
    conditional = True
    unconditional = True

    if conditional:
        if not os.path.exists(url_model_uncond):
            raise ValueError('The conditional PDM model folder does not exist (see ReadMe for pretrained models).')

    if unconditional:
        if not os.path.exists(url_model_cond):
            raise ValueError('The unconditional PDM model folder does not exist (see ReadMe for pretrained models).')

    if test_sample_DPM:

        # dpmsolver or dpmsolver++
        algorithm_type = 'dpmsolver'
        # 'singlestep', 'multistep', 'singlestep_fixed' or 'adaptive'.
        method_type = 'multistep'
        order = 3
        if unconditional:
            sample_from_ODE(
                model_path=url_model_uncond,
                clip_denoised=True,
                max_clip_val=max_clip_val,
                sample_steps=20,
                num_samples=32,
                batch_size=16,
                algorithm_type=algorithm_type,
                method_type=method_type,
                order=order,
                skip_type='logSNR',
                url_save_path='./images/examples_ODE_sampling_unconditional/',
                plot=True)
        if conditional:
            sample_from_ODE(
                model_path=url_model_cond,
                clip_denoised=True,
                max_clip_val=max_clip_val,
                sample_steps=50,
                sampled_class=8,
                num_samples=32,
                batch_size=16,
                algorithm_type=algorithm_type,
                method_type=method_type,
                order=order,
                skip_type='logSNR',
                url_save_path='./images/examples_ODE_sampling_conditional/',
                plot=True)

    if test_time_comparison:
        # Compute the list of dictionnaries to test.
        l_method_sampling = create_dict_input_ODE_method(
            l_solver=['dpmsolver'],
            l_method_type=['singlestep', 'multistep', 'adaptive'],
            l_skip_type=['logSNR'],
            l_order=[1, 2, 3])
        url_folder_samples_cond = 'images/MNIST_ODE_sampling/cond/'
        url_folder_samples_uncond = 'images/MNIST_ODE_sampling/uncond/'

        # The type of feature extractor, either 'VGG' or 'Inception'
        type_model_feature = 'VGG'
        layer_size = 2048

        if type_model_feature == 'VGG':
            url_folder_models = './models_data/MNIST_classifier_vgg_32/'
        elif type_model_feature == 'Inception':
            url_folder_models = './models_data/MNIST_classifier_Inception_32/'
        else:
            raise ValueError('The type of feature extractor is not valid.')

        if not os.path.exists(url_folder_models):
            raise ValueError(f'The folder of the feature extractor does not exist for {type_model_feature} (run example_fine_tuning.py for MNIST and {type_model_feature}).')

        if conditional:
            # Test on a conditional model
            compare_approaches_MNIST(
                type_model_feature=type_model_feature,
                url_folder_models_feature=url_folder_models,
                l_nbr_diff_steps=[20, 40, 60],
                nbr_samples=100,
                l_method_sampling=l_method_sampling,
                url_folder=url_folder_samples_cond,
                layer_size=layer_size,
                sampled_class=None,
                url_PDM_model_cond=url_model_cond,
                url_PDM_model_uncond=None)

        if unconditional:
            # Test on unconditional model
            compare_approaches_MNIST(
                type_model_feature=type_model_feature,
                url_folder_models_feature=url_folder_models,
                l_nbr_diff_steps=[20, 40, 60],
                nbr_samples=100,
                l_method_sampling=l_method_sampling,
                url_folder=url_folder_samples_uncond,
                layer_size=layer_size,
                sampled_class=None,
                url_PDM_model_cond=None,
                url_PDM_model_uncond=url_model_uncond)
