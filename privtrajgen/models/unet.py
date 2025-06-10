#!/usr/bin/env python3
"""UNet Model from https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
modified for 1D data."""
import logging
from typing import Optional, List, Sequence, Any, Union, Tuple

import math
import torch
from torch import nn

from privtrajgen.models.diffTraj.utils.Traj_UNet import get_timestep_embedding
from .unet_layers import Downsample, Upsample, Normalize, ResnetBlock, nonlinearity, make_attn

# Constants
ATTN_TYPE = "vanilla"

log = logging.getLogger()


def logc(header: str, *args: Any, column_width: int = 40) -> None:
    log.info(f"{header:<{column_width}}" + "".join([str(arg) for arg in args]))


class UNetModel(nn.Module):

    def __init__(self,
                 traj_length: int,
                 in_channels: int,  # Number of channels in the input
                 out_ch: int,  # Number of channels in the output
                 ch: int = 128,  # Base number of channels (hidden dim)
                 ch_mult: Sequence[int] = (1, 2, 2, 2),  # Channel multiplier at each resolution
                 num_res_blocks: int = 2,  # Number of residual blocks at each resolution
                 attn_resolutions: List[int] = [16],  # Resolutions to use attention layers
                 dropout: float = 0.1,  # Dropout probability
                 resamp_with_conv: bool = True,  # Use convolutions for resampling
                 use_timestep: bool = False,  # Only for Diffusion models
                 attn_type: str = ATTN_TYPE,  # Type of attention layer (vanilla, linear, etc.)
                 activation: str = "swish",  # Activation function
                 conditional: bool = False  # Conditional model
                 ):
        """
            UNet model for 1D data.

            :param traj_length: Length of the trajectory (sequence length).
            :param in_channels: Number of channels in the input.
            :param out_ch: Number of channels in the output.
            :param ch: Base number of channels (hidden dimension).
            :param ch_mult: Channel multiplier at each resolution.
            :param num_res_blocks: Number of residual blocks at each resolution.
            :param attn_resolutions: Resolutions to use attention layers.
            :param dropout: Dropout probability.
            :param resamp_with_conv: Use convolutions for resampling.
            :param use_timestep: Whether to use timestep embedding (only for diffusion models).
            :param attn_type: Type of attention layer (e.g., vanilla, linear).
            :param activation: Activation function.
            :param conditional: Whether the model is conditional.
        """
        super().__init__()
        self.ch = ch
        # temb channels is misleading because we use the same channels for the context/conditional information
        self.temb_ch = self.ch * 4 if (use_timestep or conditional) else None
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.seq_length = traj_length
        self.in_channels = in_channels
        self.activation = activation
        self.conditional = conditional

        self.use_timestep = use_timestep
        if self.use_timestep:
            log.warning("Are you sure you want to use timesteps - Only required for diffusion!")
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = traj_length
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        target_resolutions = [traj_length]
        for i_level in range(self.num_resolutions):
            target_resolutions.append(curr_res)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         activation=activation))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         activation=activation))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, target_length=target_resolutions[i_level])
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param x: Tensor of shape (batch_size, in_channels, traj_length)
        :param t: Optional diffusion timestep tensor. Only required for diffusion models.
        :param context: Optional context tensor.
        """
        # If conditional model, context tensor must be provided, but if not, it must be None
        if self.conditional == (context is None):
            raise ValueError("Context tensor must be provided for conditional models.")
        # Instead of concatenating context, we add it to the timestep embedding or use it instead of timestep embedding
        # This has the advantage that it is passed to multiple layers!
        # if context is not None:
        #     # assume aligned context, cat along channel axis
        #     x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            if t is None:
                raise ValueError("Timestep tensor `t` must be provided when `use_timestep` is True.")
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb, activation=self.activation)
            temb = self.temb.dense[1](temb)
            if context is not None:
                temb = temb + context
        else:
            temb = context

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, activation=self.activation)
        h = self.conv_out(h)
        return h


class UNetEncoder(nn.Module):
    """The encoder part of the UNet, can be used for VAE."""

    def __init__(
            self,
            traj_length: int,
            in_channels: int = 2,
            num_res_blocks: int = 2,
            attn_resolutions: List[int] = [16],
            ch: int = 128,
            latent_dim: Optional[int] = None,  # Defaults to ch * ch_mult[-1]
            ch_mult: Sequence[int] = (1, 2, 2, 2),
            dropout: float = 0.1,
            resample_via_conv: bool = True,
            vae: bool = True,  # If True, return mu and logvar, otherwise, return enc
            activation: str = "swish",
            conditional: bool = False,
    ):
        """
        :param traj_length: Length of the trajectories.
        :param in_channels: Number of input channels. Default is 2 = (lat, lon).
        :param num_res_blocks: Number of residual blocks. Default is 2.
        :param attn_resolutions: List of resolutions (=sequence lengths) at which to apply attention. Default is [16].
        :param ch: Base channel number. Default is 128.
        :param latent_dim: Latent dimension (Output dimension). Defaults to ch * ch_mult[-1].
        :param ch_mult: Tuple of channel multipliers for each resolution. Default is (1, 2, 2, 2).
        :param dropout: Dropout rate. Default is 0.1.
        :param resample_via_conv: Whether to use convolution for resampling. Default is True.
        :param vae: If True, return mu and logvar, otherwise, return enc. Default is True.
        :param activation: Activation function. Default is "swish".
        :param conditional: If True, use conditional embeddings. Default is False.
        """
        super().__init__()

        self.ch = ch  # Base channel number (hidden dim)
        self.latent_dim = latent_dim or ch * ch_mult[-1]  # Latent dimension (Output dimension)
        self.num_resolutions = len(ch_mult)  # Number of different resolutions = sequence lengths
        self.num_res_blocks = num_res_blocks  # Number of residual blocks
        self.traj_length = traj_length  # Length of the trajectories
        self.in_channels = in_channels  # Number of input channels
        self.temb_ch = 4 * self.ch if conditional else None  # Timestep & context embedding channels
        self.vae = vae  # If True, return mu and logvar, otherwise, return enc
        self.activation = activation  # Activation function
        self.conditional = conditional  # If True, use conditional embeddings

        # Downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = traj_length
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         activation=activation))
                block_in = block_out
                if curr_res in attn_resolutions:
                    # I don't think this is ever used
                    logc("Adding Attention at resolution", curr_res)
                    attn.append(make_attn(block_in, attn_type=ATTN_TYPE))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resample_via_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)
        self.mid.attn_1 = make_attn(block_in, attn_type=ATTN_TYPE)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2 * self.latent_dim if self.vae else self.latent_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the encoder.

        :param x: The input tensor
        :param context: Optional context tensor.
        """
        # If conditional model, context tensor must be provided, but if not, it must be None
        if self.conditional == (context is None):
            raise ValueError("Context tensor must be provided for conditional models.")
        temb = context

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h, activation=self.activation)
        h = self.conv_out(h)
        if self.vae:
            # Split h into mu and log_var
            mu, log_var = torch.chunk(h, 2, dim=1)
            return mu, log_var
        else:
            return h


class UNetDecoder(nn.Module):
    """The decoder part of the UNet."""

    def __init__(
            self,
            traj_length: int,
            out_ch: int = 2,
            ch: int = 128,
            num_res_blocks: int = 2,
            ch_mult: Sequence[int] = (1, 2, 2, 2),
            latent_dim: Optional[int] = None,  # Defaults to ch * ch_mult[-1]
            attn_resolutions: List[int] = [16],
            resample_via_conv: bool = True,
            dropout: float = 0.1,
            tanh_out: bool = False,
            activation: str = "swish",
            conditional: bool = False,
    ):
        super().__init__()

        self.ch = ch  # Base channel number
        self.num_resolutions = len(ch_mult)  # Number of different resolutions = sequence lengths
        self.num_res_blocks = num_res_blocks  # Number of residual blocks
        self.traj_length = traj_length  # Length of the trajectories
        self.temb_ch = 4 * self.ch if conditional else None  # Timestep & context embedding channels
        self.latent_dim = latent_dim or ch * ch_mult[-1]  # Latent dimension (Output dimension)
        self.give_pre_end = False  # If True, return h before the final layer
        self.tanh_out = tanh_out  # If True, apply tanh to the output
        self.activation = activation  # Activation function
        self.conditional = conditional  # If True, use conditional embeddings

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.input_length = self.traj_length // 2 ** (self.num_resolutions - 1)
        curr_res = self.input_length
        target_resolutions = [self.traj_length, ] + [math.ceil(self.traj_length / (2 ** i)) for i in
                                                     range(self.num_resolutions)]
        self.z_shape = (1, self.latent_dim, curr_res, curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv1d(self.latent_dim,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)
        self.mid.attn_1 = make_attn(block_in, attn_type=ATTN_TYPE)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       activation=activation)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         activation=activation))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=ATTN_TYPE))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resample_via_conv, target_length=target_resolutions[i_level])
                # curr_res = curr_res * 2
                curr_res = target_resolutions[i_level]
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # If conditional model, context tensor must be provided, but if not, it must be None
        if self.conditional == (context is None):
            raise ValueError(f"Context tensor must be provided for conditional models:"
                             f" {self.conditional} vs. {context is None}")

        # timestep & context embedding
        temb = context

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h, activation=self.activation)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
