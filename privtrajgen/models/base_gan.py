#!/usr/bin/env python3
"""BaseClass for GANs"""
import logging
import pprint
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union, List, Callable, cast, Any

import math
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pydantic import validate_call
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from privtrajgen.evaluation.config.gan_config import GANConfig
from privtrajgen.models.ema import EMAHelper
from privtrajgen.utils import nn_utils as utils, time_to_str
from privtrajgen.utils.nn_utils import compute_steps_and_epochs, get_tensorboard_writer, is_validation_step, get_device
from .cond_emb import split_batch
from .dp_wrapper import DP_Wrapper
from .save_and_load import save_model, load_model, prepare_parameter_file

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


# Determine MAX_PHYSICAL_BATCH_SIZE based on available GPU memory
available_memory = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if torch.cuda.is_available() else 0  # Convert bytes to GB
MAX_PHYSICAL_BATCH_SIZE = 1024 if available_memory > 80 else 512
logc("Available GPU Memory:", f"{available_memory:.2f} GB")
logc("MAX_PHYSICAL_BATCH_SIZE:", MAX_PHYSICAL_BATCH_SIZE)
GEN_PLACEHOLDER = "gen"
DIS_PLACEHOLDER = "dis"
COM_PLACEHOLDER = "com"


class BaseGenerator(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_noise(self, batch_size: int) -> torch.Tensor:
        """Return noise tensor that has a suitable shape for the generator.

        :param batch_size: Batch size.
        :return: Noise tensor.
        """
        pass

    @property
    def device(self) -> torch.device:
        """Assumes that all parameters are on the same device"""
        return next(self.parameters()).device


class BaseGAN(ABC, nn.Module):
    def __init__(self, name: Optional[str]):
        super().__init__()
        self.name = name
        self.gen: BaseGenerator
        self.dis: nn.Module
        if self.name:
            self.param_path: Optional[str] = prepare_parameter_file(modelname=self.name, use_model_placeholder=True)
        else:
            self.param_path = None

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def training_loop(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def device(self) -> torch.device:
        """Assumes that all parameters are on the same device"""
        return next(self.parameters()).device

    def save_parameters(self, epoch: int, save_individual: bool = False) -> None:
        if self.param_path is None:
            raise ValueError("No parameter path provided.")
        # Save individual models
        if save_individual:
            # Save generator
            save_model(self.gen, epoch=epoch, param_path=self.param_path, model_placeholder=GEN_PLACEHOLDER,
                       verbose=False)
            # Save discriminator
            save_model(self.dis, epoch=epoch, param_path=self.param_path, model_placeholder=DIS_PLACEHOLDER,
                       verbose=False)
        # Save combined model
        save_model(self, epoch=epoch, param_path=self.param_path, model_placeholder=COM_PLACEHOLDER)

    def load_parameters(self, epoch: int, param_path: Optional[str] = None) -> None:
        param_path = param_path or self.param_path

        if param_path is None:
            raise ValueError("No parameter path provided.")

        # Load combined model
        load_model(self, epoch=epoch, param_path=param_path, model_placeholder=COM_PLACEHOLDER, device=self.device)


# noinspection PyUnusedLocal
def _save_params(
        gen: DP_Wrapper,
        dis: DP_Wrapper,
        ema_gen: Optional[EMAHelper],
        ema_dis: Optional[EMAHelper],
        gan: Optional[BaseGAN],
        ema: bool,
        param_file_gen: str,
        param_file_dis: str,
        epoch: int,
) -> None:
    if gan is not None:
        gan.save_parameters(epoch=epoch, save_individual=False)
    # Save individual models
    gen.save_parameters(
        param_file=param_file_gen,
        epoch=epoch,
        ema_helper=ema_gen,
    )
    dis.save_parameters(
        param_file=param_file_dis,
        epoch=epoch,
        ema_helper=ema_dis,
    )


@validate_call(config={"arbitrary_types_allowed": True})
def train_gan(
        dataloader: DataLoader,
        config: GANConfig,
        gan: Optional[BaseGAN] = None,
        generator: Optional[BaseGenerator] = None,
        discriminator: Optional[nn.Module] = None,
        plot_fn: Optional[Callable] = None,
) -> None:
    """
    Train a GAN model.

    :param dataloader: DataLoader for the dataset.
    :param config: GANConfig object.
    :param gan: GAN model to train.
    :param generator: Generator model to train.
    :param discriminator: Discriminator model to train.
    :param plot_fn: Function to plot the results.
    """
    # Ensure that either a GAN or both gen and dis are provided, but not both
    if (gan is None) and (generator is None or discriminator is None):
        raise ValueError("Provide a GAN or both gen and dis.")
    # Assert that gan.gen == generator and gan.dis == discriminator if both are provided
    if gan is not None and generator is not None and gan.gen != generator:
        raise ValueError("Generator mismatch.")
    if gan is not None and discriminator is not None and gan.dis != discriminator:
        raise ValueError("Discriminator mismatch.")

    # Get models
    gen: BaseGenerator = cast(BaseGenerator, gan.gen if gan else generator)
    dis: nn.Module = cast(nn.Module, gan.dis if gan else discriminator)

    # Move models to device
    device = get_device(config.gpu)
    gen = gen.to(device)
    dis = dis.to(device)

    # Compute steps or epochs based on the provided value.
    config.steps, config.epochs = compute_steps_and_epochs(steps=config.steps, epochs=config.epochs,
                                                           steps_per_epoch=len(dataloader))

    if config.name is None:
        raise ValueError("Provide a name for the model.")

    # Tensorboard writer
    writer = get_tensorboard_writer(model_name=config.name, use_tensorboard=config.tensorboard)

    # Write config to tensorboard as metadata (pretty print)
    writer.add_text("Config", pprint.pformat(asdict(config)))

    # Parameter path
    param_path_g = prepare_parameter_file(modelname=config.name, use_model_placeholder=True,
                                          model_placeholder=GEN_PLACEHOLDER)
    param_path_d = prepare_parameter_file(modelname=config.name, use_model_placeholder=True,
                                          model_placeholder=DIS_PLACEHOLDER)

    # Optimizer
    opt_gen = utils.get_optimizer(parameters=gen.parameters(), optimizer=config.optimizer, lr=config.lr_gen,
                                  loss=config.loss)
    opt_dis = utils.get_optimizer(parameters=dis.parameters(), optimizer=config.optimizer, lr=config.lr_dis,
                                  loss=config.loss)

    # DP
    dis = cast(DP_Wrapper,
               DP_Wrapper(name=config.name, model=dis, optimizer=opt_dis, dp=config.dp.dp and config.dp_discriminator))
    gen = cast(DP_Wrapper,
               DP_Wrapper(name=config.name, model=gen, optimizer=opt_gen, dp=config.dp.dp and config.dp_generator))
    # Ensure the optimizers cannot be accessed directly
    del opt_gen
    del opt_dis

    # Update parameter paths
    dis.param_file = param_path_d
    gen.param_file = param_path_g
    if config.dp.dp:
        # Initialize DP
        dp_model = dis if config.dp_discriminator else gen
        dataloader = dp_model.init_dp(
            dataloader=dataloader,
            epochs=config.epochs,
            max_grad_norm=config.dp.max_grad_norm,
            target_epsilon=config.dp.target_epsilon,
            delta=config.dp.target_delta,
        )
        # Update Optimizers
        opt_gen = gen.opt
        opt_dis = dis.opt
        dp_opt = opt_dis if config.dp_discriminator else opt_gen
    else:
        dp_opt = None
        dp_model = None

    # EMA Helper
    if config.ema:
        ema_gen = EMAHelper(0.999, module=gen)
        ema_gen.register_optimizer(gen.opt)
        ema_dis = EMAHelper(0.999, module=dis)
        ema_dis.register_optimizer(dis.opt)
    else:
        ema_gen = None
        ema_dis = None

    step = 0
    start_time = datetime.now()
    n_phy_batch = math.ceil(config.batch_size / MAX_PHYSICAL_BATCH_SIZE) if config.dp.dp else 1
    for epoch in trange(1, config.epochs + 1, desc='Epochs'):
        with (
                BatchMemoryManager(
                    data_loader=dataloader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=dp_opt
                ) if (
                        config.dp.dp and MAX_PHYSICAL_BATCH_SIZE < config.training.batch_size
                ) else (
                        nullcontext(dataloader)
                )
        ) as new_dataloader:
            for batch in tqdm(new_dataloader, desc="Batches", leave=False, total=len(new_dataloader)):
                step += 1
                real_step = step // n_phy_batch

                # Move to device
                real_batch, context = split_batch(batch, config=config, device=device, dtype=torch.float32)
                batch_size = real_batch.size(0)  # Might differ from config in case of DP due to Poisson sampling

                # #############################################
                # Train Discriminator
                # #############################################
                dis.train()
                gen.train()
                dis.opt.zero_grad()

                # Real data
                d_real = dis(real_batch, context=context)
                # Fake data
                z = gen.get_noise(batch_size).to(device=device)
                fake_batch = gen(z, context=context)  # We reuse the fake_batch below for the generator's training
                d_fake = dis(fake_batch.detach(), context=context)

                # Loss
                if config.loss.wgan:
                    loss_dis_raw = -d_real.mean() + d_fake.mean()
                    if config.loss.gradient_penalty:
                        gp = utils.compute_gradient_penalty(
                            discriminator=dis, real=real_batch, synthetic=fake_batch.detach(),
                            lp=config.loss.lipschitz_penalty, context=context,
                        )
                        loss_dis = config.loss.lambda_gp * gp + loss_dis_raw
                    else:
                        loss_dis = loss_dis_raw
                else:
                    valid = torch.full_like(d_real, 0.9, device=device)
                    fake = torch.full_like(d_fake, 0.0, device=device)
                    loss_dis_raw = F.binary_cross_entropy_with_logits(d_real,
                                                                      valid) + F.binary_cross_entropy_with_logits(
                        d_fake, fake)
                    loss_dis = loss_dis_raw

                loss_dis.backward()
                if config.grad_norm_clip > 0 and not config.dp.dp and not config.dp.clip_search:
                    # Clip the gradient norm (Both DP and ClipSearch have their own clipping)
                    grad_norm = torch.nn.utils.clip_grad_norm_(dis.parameters(), max_norm=config.grad_norm_clip)
                    writer.add_scalar("Grad/DisGradNorm", grad_norm, real_step)
                dis.opt.step()  # Also updates EMA
                dis.opt.zero_grad()

                if step % n_phy_batch == 0:
                    # For WGAN, one has to plot -D_loss: https://github.com/martinarjovsky/WassersteinGAN/issues/9
                    writer.add_scalar('Loss/Dis', -loss_dis_raw, real_step)
                    if config.loss.gradient_penalty:
                        writer.add_scalar('Loss/GP', gp, real_step)

                # Clip weights for WGAN (not for WGAN-GP or WGAN-LP)
                if config.loss.wgan and not config.loss.gradient_penalty:
                    utils.clip_discriminator_weights(dis=dis, clip_value=0.01)

                # #############################################
                # Train Generator
                # #############################################
                if step % config.n_critic == 0:
                    gen.opt.zero_grad()

                    d_fake = dis(fake_batch, context=context)  # This time not detached

                    if config.loss.wgan:
                        loss_gen = -d_fake.mean()
                    else:
                        valid = torch.ones_like(d_fake, device=device)
                        loss_gen = F.binary_cross_entropy_with_logits(d_fake, valid)
                    loss_gen.backward()
                    if config.grad_norm_clip > 0 and not config.dp.dp and not config.dp.clip_search:
                        # Clip the gradient norm (Both DP and ClipSearch have their own clipping)
                        grad_norm = torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=config.grad_norm_clip)
                        writer.add_scalar("Grad/GenGradNorm", grad_norm, real_step)
                    gen.opt.step()  # Also updates EMA
                    if step % n_phy_batch == 0:
                        writer.add_scalar('Loss/Gen', loss_gen, real_step)

                # Save model(s)
                if is_validation_step(step=step, real_step=real_step, ratio=n_phy_batch, freq=config.save_freq):
                    with logging_redirect_tqdm():
                        _save_params(gen=gen, dis=dis, ema_gen=ema_gen, ema_dis=ema_dis, gan=gan, ema=config.ema,
                                     param_file_gen=param_path_g, param_file_dis=param_path_d, epoch=epoch)

                # Plot
                if (is_validation_step(step=step, real_step=real_step, ratio=n_phy_batch, freq=config.plot_freq)
                    and plot_fn is not None):
                    plot_fn(real_batch=real_batch, fake_batch=fake_batch,
                            global_step=real_step, writer=writer)
                # End of batch
            # End of BMM
        # Add Epsilon to Tensorboard
        if dp_model is not None and real_step > 2:
            writer.add_scalar("DP/Epsilon", dp_model.get_epsilon(), real_step)
        # End of epoch
    # Save final model
    _save_params(gen=gen, dis=dis, ema_gen=ema_gen, ema_dis=ema_dis, gan=gan, ema=config.ema,
                 param_file_gen=param_path_g, param_file_dis=param_path_d, epoch=config.epochs)

    # Print training duration in HH:MM:SS
    logc("Training Duration:", time_to_str(datetime.now() - start_time))

    # Write an error message if epsilon is not below the target
    if dp_model is not None and dp_model.get_epsilon() > config.dp.target_epsilon:
        log.error(
            f"Failed to reach target epsilon of {config.dp.target_epsilon}! Got {dp_model.get_epsilon()} instead.")
