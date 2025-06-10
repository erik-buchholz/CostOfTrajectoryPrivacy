import torch


def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gather consts for $t$ and reshape to feature map shape
    :param consts: (N, 1, 1)
    :param t: (N, H, W)
    :return: (N, H, W)
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def q_xt_x0(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """
    Samples x_t from the conditional distribution q(x_t | x0) according to specified diffusion process parameters.
    This function simulates one step of the forward diffusion process by adding noise to the data point x0.

    :param x0: Initial data point or batch of data points.
    :param t: Time step index or tensor of indices representing the diffusion step(s).
    :param alpha_bar: Precomputed tensor of the cumulative product of alpha values across time steps.
    :return: A tuple containing:
        - x_t: The noised data point at time t: $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \text{noise}$.
        - noise: Random Gaussian noise sampled from $\mathcal{N}(0, 1)$ used in the calculation of x_t.

    Example usage:
        xt, noise = sample_xt(x0, t, alpha_bar)
    """
    # Calculate the mean and variance of x_t given x0
    mean = torch.sqrt(gather(alpha_bar, t)) * x0  # mean = \sqrt{\bar{\alpha}_t} \cdot x_0
    var = 1 - gather(alpha_bar, t)  # var = 1 - \bar{\alpha}_t
    # Sample x_t from q(xt | x0)
    noise = torch.randn_like(x0).to(device=x0.device, dtype=x0.dtype)  # noise \sim \mathcal{N}(0, 1)
    xt = mean + torch.sqrt(var) * noise
    # x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \text{noise}
    return xt, noise


def remove_noise(xt: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """
    Estimate the original data point x0 from x_t by removing injected noise.

    :param xt: The noisy data point or tensor of data points at time step t.
    :param noise: The noise component added to x0 to generate xt.
    :param t: The time step index or tensor of time steps.
    :param alpha_bar: A precomputed tensor of cumulative product of alphas over time steps.
    :return: The estimated original data point x0.
    """
    alpha_sqrt = torch.sqrt(gather(alpha_bar, t))
    one_minus_alpha_sqrt = torch.sqrt(1 - gather(alpha_bar, t))
    x0 = (xt - one_minus_alpha_sqrt * noise) / alpha_sqrt
    return x0


def compute_alpha_bar(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Computes alpha_bar (!) $\bar{\alpha}}_t$ for a given time step t and beta values.
    :param beta: The beta values used in the diffusion process.
    :param t: The time step index or tensor of time steps.
    :return: The cumulative product of alpha values up to time step t: $\bar{\alpha}_t = \prod_{i=0}^{t} (1 - \beta_i)$.
    """
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt: torch.Tensor, predicted_noise: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor, beta: torch.Tensor,
         eta: float = 0.0) -> torch.Tensor:
    """
    Computes the next state x_t_next in the reverse diffusion process, using the current state x_t,
    noise, and transition parameters.

    :param xt: The current state tensor of data points at time step t.
    :param predicted_noise: The noise tensor that was predicted by the trained model at time step t.
    :param t: The current time step index as a tensor.
    :param next_t: The next time step index as a tensor.
    :param beta: A tensor of beta values used in the diffusion process.
    :param eta: A scaling factor for adjusting the noise terms in the reverse process, default is 0.
    :return: The next state tensor x_t_next for the reverse diffusion process at time step next_t.
    """
    # Compute alpha values for the current and next time steps
    at = compute_alpha_bar(beta.to(device=xt.device), t.long())  # \bar{\alpha}_t
    at_next = compute_alpha_bar(beta, next_t.long())  # \bar{\alpha}_{t-step}

    # Calculate the denoised data point x0_t at time t
    x0_t = (xt - predicted_noise * torch.sqrt(1 - at)) / torch.sqrt(at)
    # x_0^t = \frac{x_t - \epsilon_{\theta}(x_t) \cdot \sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}

    # Compute coefficients for the Gaussian variables in the reverse process
    c1 = eta * torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))  # 0 for eta=0
    c2 = torch.sqrt((1 - at_next) - c1 ** 2)  # \sqrt{1 - \bar{\alpha}_{t-step}} for eta=0

    # Generate new Gaussian noise for the reverse step
    eps = torch.randn(xt.shape, device=xt.device)  # \epsilon \sim \mathcal{N}(0, 1)

    # Compute the next state x_t_next using the reverse process equation
    xt_next = torch.sqrt(at_next) * x0_t + c1 * eps + c2 * predicted_noise
    # For eta=0: $x_{t-\text{step}} = \sqrt{\bar{\alpha}_{t-\text{step}}} \cdot x_0^t + \sqrt{1 - \bar{\alpha}_{t-step}} \cdot \epsilon_{\theta}(x_t)$
    return xt_next


def make_beta_schedule(schedule: str = 'linear', n_timesteps: int = 1000, start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas
