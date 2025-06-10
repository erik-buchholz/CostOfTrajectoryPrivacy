#!/usr/bin/env python3
"""A VAE based on DiffTraj's UNet architecture and
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py."""
import logging
import pprint
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Tuple, Any

import math
import pytorch_warmup as warmup
import torch
from matplotlib import pyplot as plt
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pydantic import validate_call
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from privtrajgen.config import NORMALISED_BBOX
from privtrajgen.evaluation.config import UNetVAEConfig, VAEConfig
from privtrajgen.metrics import sliced_wasserstein_distance, JS_from_points
from privtrajgen.utils import time_to_str
from privtrajgen.utils.nn_utils import get_optimizer, compute_kl_weight, compute_steps_and_epochs, \
    get_tensorboard_writer, get_lr_scheduler, is_validation_step, get_device
from .base_vae import BaseVAE
from .cond_emb import ConditionalEmbedding, split_batch
from .dp_wrapper import DP_Wrapper
from .ema import EMAHelper
from .save_and_load import prepare_parameter_file
from .unet import UNetEncoder, UNetDecoder

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


# Constants
ATTN_TYPE = "vanilla"
# MAX_PHYSICAL_BATCH_SIZE = 768  # Server Yennefer (empirical value)
MAX_PHYSICAL_BATCH_SIZE = 1000  # HPC (empirical value)

class Encoder(nn.Module):
    """Encoder for the UNetVAE."""

    def __init__(self, config: UNetVAEConfig):
        super().__init__()
        self.traj_length = config.data.traj_length
        self.in_channels = config.in_channels
        self.ch = config.hidden_dim
        self.conditional = config.conditional

        self.encoder = UNetEncoder(traj_length=self.traj_length,
                                   in_channels=self.in_channels,
                                   ch=self.ch,
                                   vae=True,
                                   conditional=self.conditional)
        if self.conditional:
            self.cond_emb = ConditionalEmbedding(config=config)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)
        else:
            context = None
        mu, logvar = self.encoder(x, context=context)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder for the UNetVAE."""

    def __init__(self, config: UNetVAEConfig):
        super().__init__()
        self.traj_length = config.data.traj_length
        self.ch = config.hidden_dim
        self.out_ch = config.out_channels
        self.conditional = config.conditional

        self.decoder = UNetDecoder(traj_length=config.data.traj_length,
                                   ch=config.hidden_dim,
                                   out_ch=config.out_channels,
                                   conditional=config.conditional,
                                   tanh_out=True,
                                   )

        self.latent_dim = self.decoder.latent_dim
        self.input_length = self.decoder.input_length

        if self.conditional:
            self.cond_emb = ConditionalEmbedding(config=config)

    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.conditional == (context is None):
            raise ValueError("Conditional flag and attribute tensor must match.")
        if self.conditional:
            context = self.cond_emb(context)
        else:
            context = None
        return self.decoder(z, context=context)


class UNetVAE(BaseVAE):
    def plot_gradients(self, writer: SummaryWriter, global_step: int) -> None:
        pass

    def __init__(
            self,
            config: UNetVAEConfig,
    ):
        super().__init__(name=config.name or "UNetVAE")

        # Save conditional flag
        self.conditional = config.conditional
        logc("Conditional Model:", config.conditional)

        # Define the encoder and decoder
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)

        # Define the reconstruction loss. Default is MSE Loss. TrajLoss can be used as alternative.
        self.reconstruction_loss = torch.nn.MSELoss()

        # Determine optimizer
        self.opt = get_optimizer(self.parameters(), optimizer=config.optimizer)

        # Log parameter counts for both models
        total_param_count = sum(p.numel() for p in self.parameters())
        enc_param_count = sum(p.numel() for p in self.encoder.parameters())
        dec_param_count = sum(p.numel() for p in self.decoder.parameters())
        logc("Total parameters:", f"{total_param_count:,}")
        logc("Encoder parameters:", f"{enc_param_count:,}")
        logc("Decoder parameters:", f"{dec_param_count:,}")
        logc("Parameter Ration [D/E]:", f"{dec_param_count / enc_param_count:.2f}")

    def sample(self, batch_size: Optional[int] = None, length: Optional[int] = None, z: Optional[torch.Tensor] = None,
               *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        decoded output.
        :param batch_size: (int) Batch size for sampling
        :param z: (torch.Tensor) Latent space tensor
        :param length: (int) Length of the output
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return:
        """
        assert length is None or length == self.decoder.input_length, (f"UNeTVAE only supports fixed length"
                                                                       f" {self.decoder.input_length}.")
        if z is None:
            z = torch.randn(batch_size, self.decoder.latent_dim, self.decoder.input_length).to(device=self.device)
        else:
            # Test if z has the correct shape
            if z.shape[1:] != (self.decoder.latent_dim, self.decoder.input_length):
                raise ValueError(
                    f"z has shape {z.shape}, but expected {self.decoder.latent_dim, self.decoder.input_length}.")
        samples = self.decode(z, *args, **kwargs)
        return samples

    @validate_call(config={"arbitrary_types_allowed": True})
    def training_loop(self, dataloader: DataLoader, config: VAEConfig) -> None:
        """
        Training loop for the VAE

        :param dataloader: DataLoader for the training data
        :param config: Configuration for the VAE
        :return:
        """
        assert isinstance(config, UNetVAEConfig), f"Expected UNetVAEConfig, but got {type(config)}."
        device = get_device(config.gpu)
        self.to(device=device)

        # Compute steps or epochs based on the provided value.
        config.steps, config.epochs = compute_steps_and_epochs(steps=config.steps, epochs=config.epochs,
                                                               steps_per_epoch=len(dataloader))

        # Make sure optimizer has been defined
        if self.opt is None:
            raise ValueError("Optimizer has to be defined in the model as self.opt.")
        # Verify that optimizer uses config.optimizer.lr
        if self.opt.param_groups[0]['lr'] != config.optimizer.lr:  # type: ignore
            raise ValueError(
                f"Optimizer has learning rate {self.opt.param_groups[0]['lr']} but expected {config.optimizer.lr}.")

        # Update name & parameter path
        config.name = config.name or self.name
        self.name = config.name
        self.param_path = prepare_parameter_file(modelname=self.name)

        # Initialize DP [The DP Wrapper can even be used without DP]
        model = DP_Wrapper(name=self.name, model=self, optimizer=self.opt, dp=config.dp.dp)
        if config.dp.dp:
            dataloader = model.init_dp(dataloader=dataloader, epochs=config.epochs,
                                       max_grad_norm=config.dp.max_grad_norm,
                                       target_epsilon=config.dp.target_epsilon, delta=config.dp.target_delta)
        # Replace the non-DP optimizer in case it was changed
        self.opt = model.opt

        # Define a lr scheduler
        model.lr_scheduler = get_lr_scheduler(opt=self.opt, lr_scheduler=config.lr_scheduler,
                                              steps=config.steps, steps_per_epoch=len(dataloader))

        # Define warmup scheduler
        model.warmup_scheduler = warmup.LinearWarmup(model.opt, warmup_period=config.warmup_steps)

        # Print the used KL Annealing Technique
        logc(f"KL Annealing:", config.kl_annealing)

        # Create Tensorboard connection
        writer = get_tensorboard_writer(model_name=model.name, use_tensorboard=config.tensorboard)

        # Write config to tensorboard as metadata
        writer.add_text("Config", pprint.pformat(asdict(config)))

        if config.ema:
            ema_helper = EMAHelper(config.ema_rate, module=model)
            ema_helper.register_optimizer(model.opt)
            if config.ema_rate > 0.999:
                log.warning(f"Unusually high EMA rate for VAE: {config.ema_rate}")
        else:
            ema_helper = None
        logc("Using EMA:", config.ema)

        step, real_step = 0, 0
        n_phy_batch = math.ceil(config.batch_size / MAX_PHYSICAL_BATCH_SIZE) if config.dp.dp else 1
        start_time = datetime.now()
        for epoch in trange(1, config.epochs + 1, desc='Epochs'):
            with (
                    BatchMemoryManager(
                        data_loader=dataloader,
                        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                        optimizer=model.opt
                    ) if (
                            model.dp_initialised and MAX_PHYSICAL_BATCH_SIZE < config.training.batch_size
                    ) else (
                            nullcontext(dataloader)
                    )
            ) as new_dataloader:
                for i, batch in enumerate(tqdm(
                        new_dataloader, desc='Batches', leave=False, total=len(new_dataloader))):
                    step += 1
                    real_step = step // n_phy_batch  # Needed if BMM used

                    samples, attr = split_batch(batch, config=config, device=device, dtype=torch.float32)

                    # Zero the gradients
                    model.train()
                    model.opt.zero_grad()
                    # Forward pass
                    output, input, mu, logvar = model(samples, attr)
                    # Compute KL Weight
                    kl_weight = compute_kl_weight(real_step,
                                                  total_steps=config.steps,
                                                  min_weight=config.kl_annealing.klw_min,
                                                  max_weight=config.kl_annealing.klw_max,
                                                  type=config.kl_annealing.anneal_type)
                    # Compute loss
                    loss, rec_loss, kl_div = self.loss_function(output, samples, mu, logvar, kl_weight=kl_weight)
                    # Backward pass
                    loss.backward()
                    model.opt.step()

                    # Update warmup
                    with model.warmup_scheduler.dampening():
                        if model.lr_scheduler is not None and real_step >= config.warmup_steps:
                            # Ignore warmup for the sake of the LR Scheduler
                            model.lr_scheduler.step()

                    if is_validation_step(step=step, real_step=real_step, ratio=n_phy_batch, freq=100):
                        writer.add_scalar("Loss/Loss", loss.item(), global_step=real_step)
                        writer.add_scalar("Loss/Rec", rec_loss.item(), global_step=real_step)
                        writer.add_scalar("Loss/KL Div", kl_div.item(), global_step=real_step)
                        writer.add_scalar("Param/KL Weight", kl_weight, global_step=real_step)
                        writer.add_scalar("Param/Learning Rate", model.opt.param_groups[0]['lr'], global_step=real_step)
                        self.plot_gradients(writer, global_step=real_step)
                        # Validation Loss - Another Forward Pass in eval mode
                        with torch.no_grad():
                            model.eval()
                            output, input, mu, logvar = model(samples, attr)
                            loss, rec_loss, kl_div = self.loss_function(output, samples, mu, logvar,
                                                                        kl_weight=kl_weight)
                            writer.add_scalar("Loss/Validation", loss.item(), global_step=real_step)
                            # writer.add_scalar("Loss/Val Rec", rec_loss.item(), global_step=real_step)
                            # writer.add_scalar("Loss/Val KL Div", kl_div.item(), global_step=real_step)

                    # Save model every X steps
                    if is_validation_step(step=step, real_step=real_step, ratio=n_phy_batch,
                                          freq=config.save_freq):
                        with logging_redirect_tqdm():
                            model.save_parameters(epoch=epoch, ema_helper=ema_helper)

                    if is_validation_step(step=step, real_step=real_step, ratio=n_phy_batch,
                                          freq=config.plot_freq):
                        tmp = []
                        for title, data in [("Original", samples), ("Reconstruction", output),
                                            ("Generated", model.sample(samples.size(0), context=attr))]:
                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            points = data.transpose(1, 2).reshape(-1, 2).detach().cpu().numpy()
                            tmp.append(points)
                            ax.scatter(points[:, 0], points[:, 1], color='blue', alpha=0.1, s=0.1)
                            writer.add_figure(f"{title}", fig, global_step=real_step)

                        # Add some metrics to tensorboard
                        points_orig, _, points_syn = tmp
                        swd = sliced_wasserstein_distance(points_orig, points_syn, seed=config.seed)
                        writer.add_scalar("Metrics/SWD", swd, real_step)
                        jsd = JS_from_points(points_orig, points_syn, grid_size=32, bbox=NORMALISED_BBOX)
                        writer.add_scalar("Metrics/JSD", jsd, real_step)
                    # End of batch loop
                # End of BMM with statement
            # End of epoch loop
        # Save model at the end of training
        model.save_parameters(epoch=config.epochs, ema_helper=ema_helper)
        # Print the time needed for training in HH:MM:SS
        logc("Total Training Duration:", time_to_str(datetime.now() - start_time))
        # Write an error message if epsilon is not below the target
        if config.dp.dp and model.get_epsilon() > config.dp.target_epsilon:
            log.error(
                f"Failed to reach target epsilon of {config.dp.target_epsilon}! Got {model.get_epsilon()} instead.")


def get_unet_vae_model(config: UNetVAEConfig) -> UNetVAE:
    model = UNetVAE(config)
    return model
