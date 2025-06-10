args = {
    'data': {
        'dataset': '',
        'traj_path1': '/xxxxxxx',
        'head_path2': '/xxxxxxx',
        'traj_length': 200,
        'channels': 2,
        'uniform_dequantization': False,
        'gaussian_dequantization': False,
        'num_workers': True,
    },
    'model': {
        'type': "simple",
        'attr_dim': 0,
        'guidance_scale': 0,
        'in_channels': 2,
        'out_ch': 2,
        'ch': 128,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.1,
        'var_type': 'fixedlarge',
        'ema_rate': 0.9999,
        'ema': True,
        'resamp_with_conv': True,
    },
    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        'batch_size': 10240,  # For DP-SGD, the larger the batch the better
        # We don't have to worry about the number of epochs, since we have n_iters
        # 'n_epochs': 200,  # The original datasets were much larger. 200 epochs equal approx. 500k iterations
        'n_iters': 5000,  # Number of iterations, takes precedence over n_epochs
        'save_freq': 1000,  # Save frequency in steps (= Gradient updates or real steps, not physical batches)
        'validation_freq': 500,  # Plotting frequency in steps (= Gradient updates or real steps, not physical batches)
        'lr': 5e-4,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    },
    'dp': {
        'dp': True,
        'target_delta': None,  # Set delta to 1/n^1.1 by default
        'target_epsilon': 10.0,
        'max_grad_norm': 0.1,
        'dp_accountant': 'prv',
        'clip_search': False,
        'adaptive_grad_norm': True,
    }
}
