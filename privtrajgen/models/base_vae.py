#!/usr/bin/env python3
"""
Base model for VAE architectures
Source: https://github.com/AntixK/PyTorch-VAE
"""
import logging
from abc import abstractmethod
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pytorch_warmup as warmup
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from .base_model import BaseModel
from .dp_wrapper import DP_Wrapper
from .save_and_load import prepare_parameter_file
from privtrajgen import config as cfg
from privtrajgen.config import VAEType
from privtrajgen.datasets.mnist_data import show_mnist_samples
from privtrajgen.evaluation.config import VAEConfig
from privtrajgen.utils import nn_utils, visualise
from privtrajgen.utils.nn_utils import NullWriter, compute_kl_weight, compute_steps_and_epochs

log = logging.getLogger()

def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class BaseVAE(BaseModel):

    def __init__(self,
                 name: str,
                 param_dir: Optional[str] = None,
                 # Loss Parameters
                 loss_type: VAEType = VAEType.VANILLA,
                 beta: float = 4.0,
                 gamma: float = 10.0,
                 max_capacity_beta_vae: float = 25.0,
                 C_stop_iter: int = int(1e5),
                 ) -> None:
        """
        :param name: Name of the model, used for saving and logging
        :param param_dir: Path to save the model parameters
        # Beta-VAE Parameters
        :param beta: Beta value, used by BetaVAE loss. Default: 4.0 as in the original paper
        :param gamma: Gamma value, used by BetaVAE+ loss
        :param max_capacity_beta_vae: Max capacity, used by BetaVAE+ loss
        :param C_stop_iter: Number of iterations to stop increasing the capacity, used by BetaVAE+ loss
        """
        # Call Superclass
        super().__init__(name=name, param_dir=param_dir)

        if loss_type != VAEType.VANILLA:
            log.warning(
                f"Are you sure you want to use {loss_type} loss? "
                "You can simply set the KL Weight to Beta instead."
            )
        # Beta-VAE Parameters
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.C_max = torch.Tensor([max_capacity_beta_vae])
        self.C_stop_iter = C_stop_iter
        self.num_iter = 0
        # Optimizer needs to be defined in the subclass, as it requires the model parameters
        self.opt = None

    def encode(self, x: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input data
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: (Tuple[Tensor, Tensor) List of latent codes: [mu, log_var]
        """
        encoding: Tuple[Tensor, Tensor] = self.encoder(x, *args, **kwargs)
        return encoding

    def decode(self, z: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Maps the given latent codes
        onto the output space.
        :param z: (Tensor) Latent code
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: (Tensor) Decoded output
        """
        output = self.decoder(z, *args, **kwargs)
        return output

    def sample(self, batch_size: int, length: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        decoded output.
        :param batch_size: (int) Batch size for sampling
        :param length: (int) Length of the sequence to generate
        :return:
        """
        z = torch.randn(batch_size, length, self.latent_dim, device=self.device)
        samples = self.decode(z)
        return samples

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the VAE, which includes encoding and decoding
        :param x: (Tensor) Input data w/ shape (# features, batch_size, seq_len, feature_len)
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: List[List[Tensor]]: List of decoded features, x, mu, log_var
        """
        mu, log_var = self.encode(x, *args, **kwargs)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, *args, **kwargs), x, mu, log_var

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Re-parameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return: (Tensor) Sample from the latent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(
            self,
            generated: Union[Sequence[Tensor], Tensor],
            input: Union[Sequence[Tensor], Tensor],
            mu: Tensor,
            log_var: Tensor,
            kl_weight: float = 1.0,
            lengths: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the loss function of the VAE
        :param generated: (List[Tensor]) The generated data
        :param input: (List[Tensor])The real input data
        :param mu: (Tensor) Mean of the latent Gaussian
        :param log_var: (Tensor) Log variance of the latent Gaussian
        :param kl_weight: (float) Weight for the KL Divergence
        :param lengths: (List[int]) Length of the (real) sequences --> For masking
        :return: (Tensor, Tensor, Tensor) Loss value, Reconstruction loss, KL Divergence
        """
        # Reconstruction loss
        if lengths is None:
            rec_loss = self.reconstruction_loss(generated, input)
        else:
            rec_loss = self.reconstruction_loss(generated, input, lengths=lengths)

        # KL Divergence
        # We sum over the latent dimension but compute the mean over both batch and sequence length.
        # This results in a comparable loss to the reconstruction loss using MSELoss.
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        if self.loss_type == 'Vanilla':
            # Vanilla VAE
            loss = rec_loss + kl_weight * kl_div
        elif self.loss_type == 'Beta':  # https://openreview.net/forum?id=Sy2fzU9gl
            if kl_weight != 1.0:
                log.warning(f"Using Beta-VAE with KL Weight {kl_weight}. Are you sure?")
            loss = rec_loss + self.beta * kl_weight * kl_div
        elif self.loss_type == 'Beta+':  # https://arxiv.org/pdf/1804.03599.pdf
            self.num_iter += 1
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = rec_loss + self.gamma * kl_weight * (kl_div - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss, rec_loss, kl_div

    @abstractmethod
    def plot_gradients(self, writer: SummaryWriter, global_step: int) -> None:
        pass

    def training_loop(self,
                      dataloader: DataLoader,
                      config: VAEConfig,
                      ) -> None:
        """
        Training loop for the VAE

        :param dataloader: DataLoader for the training data
        :param config: Configuration for the training
        """
        device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Compute steps or epochs based on the provided value. 'steps' takes precedence.
        compute_steps_and_epochs(steps=config.steps, epochs=config.epochs, steps_per_epoch=len(dataloader))

        # Make sure optimizer has been defined
        if self.opt is None:
            raise ValueError("Optimizer has to be defined in the model as self.opt.")
        # Verify that optimizer uses config.optimizer.lr
        if self.opt.param_groups[0]['lr'] != config.optimizer.lr:  # type: ignore
            raise ValueError(f"Optimizer has learning rate {self.opt.param_groups[0]['lr']} "
                             f"but expected {config.optimizer.lr}.")

        if config.dataset.upper() not in self.name.upper():
            self.name += f"_{config.dataset.upper()}"
            config.name = self.name
        # Update parameter path
        self.param_path = prepare_parameter_file(modelname=self.name)

        # Initialize DP [The DP Wrapper can even be used without DP]
        model = DP_Wrapper(name=self.name, model=self, optimizer=self.opt, dp=config.dp.dp)
        if config.dp.dp:
            dataloader = model.init_dp(dataloader=dataloader, epochs=config.epochs,
                                       max_grad_norm=config.dp.max_grad_norm,
                                       noise_multiplier=config.dp.target_epsilon, delta=config.dp.target_delta)
        # Replace the non-DP optimizer in case it was changed
        self.opt = model.opt

        # Define a lr scheduler
        model.lr_scheduler = nn_utils.get_lr_scheduler(opt=self.opt, lr_scheduler=config.lr_scheduler,
                                                       steps=config.steps, steps_per_epoch=len(dataloader))

        # Define warmup scheduler
        model.warmup_period = config.warmup_steps
        model.warmup_scheduler = warmup.LinearWarmup(model.opt, warmup_period=model.warmup_period)

        # Print the used KL Annealing Technique
        logc(f"KL Annealing:", config.kl_annealing)

        # Create Tensorboard connection
        writer = SummaryWriter(cfg.TENSORBOARD_DIR + datetime.now().strftime('%b-%d_%X_') + model.name) if config.tensorboard else NullWriter()

        # Define a large set of real trajectories for plotting
        NUM = 1000
        # noinspection PyUnresolvedReferences
        if hasattr(dataloader.dataset, 'features') and 'latlon' in dataloader.dataset.features:
            # The batch size is too small to see anything, therefore, we randomly chose
            # NUM samples from the dataset
            real_trajs = [dataloader.dataset[x] for x in np.random.randint(0, len(dataloader.dataset), NUM)]
            # Create a padded batch
            real_traj_batch = dataloader.collate_fn(real_trajs)
            # Move to device
            real_traj_batch["samples"] = [feature.to(dtype=torch.float, device=model.device) for feature in real_traj_batch["samples"]]
        else:
            real_trajs, real_traj_batch = None, None

        step = -1
        for epoch in trange(1, config.epochs + 1, desc='Epochs'):
            for i, batch in enumerate(tqdm(dataloader, desc='Batches', leave=False)):
                step += 1

                samples: List[torch.Tensor] = batch['samples']
                lengths: List[int] = batch['lengths']

                # Move data to device.
                samples = [feature.to(dtype=torch.float, device=model.device) for feature in samples]

                model.train()
                model.opt.zero_grad()
                if lengths:
                    output, input, mu, logvar = model(samples, lengths=lengths)
                else:
                    output, input, mu, logvar = model(samples)
                # Compute KL Weight
                kl_weight = compute_kl_weight(step, total_steps=config.steps, min_weight=config.kl_annealing.klw_min,
                                              max_weight=config.kl_annealing.klw_max,
                                              type=config.kl_annealing.anneal_type)
                loss, rec_loss, kl_div = self.loss_function(output, samples, mu, logvar, kl_weight=kl_weight,
                                                            lengths=lengths)
                loss.backward()
                model.opt.step()
                # Update warmup
                with model.warmup_scheduler.dampening():
                    if model.lr_scheduler is not None and step >= model.warmup_period:
                        # Ignore warmup for the sake of the LR Scheduler
                        model.lr_scheduler.step()

                if i % 100 == 0:
                    writer.add_scalar("Loss/Loss", loss.item(), global_step=step)
                    writer.add_scalar("Loss/Rec", rec_loss.item(), global_step=step)
                    writer.add_scalar("Loss/KL Div", kl_div.item(), global_step=step)
                    writer.add_scalar("Loss/KL Weight", kl_weight, global_step=step)
                    writer.add_scalar("Param/Learning Rate", model.opt.param_groups[0]['lr'], global_step=step)
                    self.plot_gradients(writer, global_step=step)


                # noinspection PyUnresolvedReferences
                if step % config.plot_freq == 0:
                    # noinspection PyUnresolvedReferences
                    if 'mnist' in config.dataset.lower():
                        dim = self.vocab_size['mnist']
                        # Plot 25 generated samples at the end of each epoch
                        # These samples are drawn from the standard normal distribution and not from mu/logvar
                        generated = self.sample(25, 28 * 28 // dim).reshape(-1, 1, 28, 28)
                        fig, ax = show_mnist_samples(generated)
                        # Add Figure to Tensorboard
                        writer.add_figure("Samples/Generated", fig, global_step=epoch)

                        # Now, we would like to compare these samples with samples generated by sampling from
                        # the normal distribution with mean mu and variance logvar.
                        # We will generate 25 samples
                        # noinspection PyUnboundLocalVariable
                        z = self.reparameterize(mu[:25], logvar[:25])
                        reconst = self.decode(z).reshape(25, 1, 28, 28)
                        fig, ax = show_mnist_samples(reconst)
                        writer.add_figure("Samples/Reconstructed", fig, global_step=epoch)
                    # noinspection PyUnresolvedReferences
                    elif 'latlon' in dataloader.dataset.features:
                        # Plot Point clouds of the generated trajectories
                        LEN = 28
                        generated = self.sample(NUM, LEN)[0].cpu().detach().numpy().reshape(-1, 2)
                        # log.info(f"Generated Shape: {generated.shape}")
                        fig, _ = visualise.plot_pointclouds(generated, title="Generated Trajectories")
                        writer.add_figure("Samples/Generated", fig, global_step=step)

                        # Now, we would like to compare these samples with samples generated by sampling from
                        # the normal distribution with mean mu and variance logvar.
                        # z = self.reparameterize(mu, logvar)
                        # reconstructed = self.decode(z)[0]
                        reconstructed, _, _, _ = self.forward(real_traj_batch["samples"], lengths=real_traj_batch["lengths"])
                        # log.info(f"Reconstructed Shape: {reconstructed[0].shape}, {sum(real_traj_batch['lengths'])}")
                        # Remove padding
                        reconstructed = reconstructed.cpu().detach().numpy()
                        reconstructed = [x[:l] for x, l in zip(reconstructed[0], real_traj_batch["lengths"])]
                        # Reshape
                        reconstructed = np.concatenate([x.reshape(-1, 2) for x in reconstructed])
                        # log.info(f"Reconstructed Shape: {reconstructed.shape}")
                        fig, _ = visualise.plot_pointclouds(reconstructed, title="Reconstructed Trajectories")
                        writer.add_figure("Samples/Reconstructed", fig, global_step=step)

                        # Lastly, plot some of the input trajectories (Only once)
                        if step == 0:
                            # Reshape the data. First item is the encoded data, second is labels
                            original = np.concatenate([x[0].reshape(-1, 2) for x in real_trajs])
                            # Plotting a Batch, only works for large batches:
                            # original = samples[0].cpu().detach().numpy().reshape(-1, 2)
                            fig, _ = visualise.plot_pointclouds(original, title="Input Trajectories")
                            writer.add_figure("Samples/Input", fig, global_step=step)
                            # Close ALL open figures
                            plt.close('all')
