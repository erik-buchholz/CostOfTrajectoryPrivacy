import logging
from types import SimpleNamespace
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from privtrajgen.evaluation.config.difftraj_config import DiffTrajConfig
from privtrajgen.models.cond_emb import ConditionalEmbedding
from privtrajgen.models.diffTraj.configs.config import args as default_args

log = logging.getLogger(__name__)


def get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class WideAndDeep(nn.Module):
    # # Constant formats for certain conditional datasets
    # # 1. DiffTraj
    # DIFFTRAJ_CONDITIONAL_FORMAT = {
    #     "continuous_idx": slice(1, 6),
    #     "categorical_idx": [0, 6, 7],
    #     "categorical_dims": [288,257,257],
    #     "embedding_dim": 512,
    # }
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256):
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(257, hidden_dim)
        self.eid_embedding = nn.Embedding(257, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim * 3, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr: Tensor) -> Tensor:
        # Continuous attributes
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(
        ), attr[:, 6].long(), attr[:, 7].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed


def nonlinearity(x: Tensor) -> Tensor:
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.1,
                 temb_channels: int = 512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_


class Model(nn.Module):
    def __init__(self, config: Union[SimpleNamespace, DiffTrajConfig]):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(
            config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.traj_length
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in: int = 0
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
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
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

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
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: Tensor, t: Tensor, extra_embed: Tensor = None) -> Tensor:
        assert x.shape[2] == self.resolution, f"Expected {self.resolution} time steps but got {x.shape[2]}."

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

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
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Guide_UNet(nn.Module):
    """
    This is the model with conditional information.
    """

    def __init__(self, config: Union[SimpleNamespace, DiffTrajConfig]):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = ConditionalEmbedding(config=config)  # type: ignore
        self.place_emb = ConditionalEmbedding(config=config)  # type: ignore

    def forward(self, x: Tensor, t: Tensor, context: Tensor) -> Tensor:
        guide_emb = self.guide_emb(context)  # Conditional embedding [x_0^c]
        place_vector = torch.zeros(context.shape, device=context.device)

        # Place embedding is a not a great naming choice.
        # This is supposed to be the "unconditional" embedding which is used to control how deterministic the model is
        place_emb = self.place_emb(place_vector)  # Unconditional embedding = 0
        cond_noise = self.unet(x, t, guide_emb)  # ε(x_st | x_0^c) Conditional noise
        uncond_noise = self.unet(x, t, place_emb)  # ε(x_st | Θ) Unconditional noise

        # Note: The following formula refers to Equation 7 in the paper
        # ε_θ = (1 + ω) * ε(x_st | x_0^c) - ω * ε(x_st | Θ) = ε(x_st | x_0^c) + ω * (ε(x_st | x_0^c) - ε(x_st | Θ))
        pred_noise = cond_noise + self.guidance_scale * (cond_noise - uncond_noise)

        return pred_noise

    def track_gradients(self, writer: SummaryWriter, step: int) -> None:
        """Track individual gradients of the model."""
        self.guide_emb.track_gradients(writer, step, prefix='GUD')
        self.place_emb.track_gradients(writer, step, prefix='PLC')


############################ Added by Erik ############################

class Uncond_UNet(nn.Module):
    """
    This is the model without conditional information.
    """

    def __init__(self, config: Union[SimpleNamespace, DiffTrajConfig]):
        super().__init__()
        self.config = config
        self.unet = Model(config)

        # # Place embedding does not depend on the attributes
        # self.ch = config.model.ch * 4
        # self.attr_dim = config.model.attr_dim
        # self.place_emb = WideAndDeep(self.ch)

    # noinspection PyUnusedLocal
    def forward(self, x: Tensor, t: Tensor, context: Tensor = None) -> Tensor:
        """

        :param x:
        :param t:
        :param context: Ignored, only for compatibility
        :return:
        """
        # Optional place embedding
        # Author confirmed that we don't need the place embedding
        # https://github.com/Yasoz/DiffTraj/issues/11
        # batch_size = x.shape[0]
        # place_vector = torch.zeros((batch_size, 8), device=x.device)
        # place_emb = self.place_emb(place_vector)
        # uncond_noise = self.unet(x, t, place_emb)

        pred_noise = self.unet(x, t, extra_embed=None)
        return pred_noise


def get_difftraj_model(
        config: Union[SimpleNamespace, DiffTrajConfig],
        conditional: Optional[bool] = None,
) -> Union[Guide_UNet, Uncond_UNet]:
    """
    Create a model from the config.

    :param config: Configuration
    :param conditional: Whether the model uses conditional information. If None, it is inferred from the config.
    :return: Model
    """
    # Assert that conditional parameter matches config if present
    if conditional is None:
        conditional = config.conditional
    elif conditional is not None and hasattr(config, 'conditional') and conditional != config.conditional:
        raise ValueError(f"Conditional parameter {conditional} does not match config {config.conditional}.")
    # Set up the model if not provided
    model: Union[Guide_UNet, Uncond_UNet]
    if conditional:
        model = Guide_UNet(config)
    else:
        model = Uncond_UNet(config)
    return model


if __name__ == '__main__':
    temp = {}
    for k, v in default_args.items():
        temp[k] = SimpleNamespace(**v)  # type: ignore

    config = SimpleNamespace(**temp)

    batch_size = 10
    seq_len = 200
    features = 2

    t = torch.randn(batch_size)

    # Create the attribute vector, elsewhere called "head"
    depature = torch.zeros(batch_size)
    avg_dis = torch.zeros(batch_size)
    avg_speed = torch.zeros(batch_size)
    total_dis = torch.zeros(batch_size)
    total_time = torch.zeros(batch_size)
    total_len = torch.zeros(batch_size)
    sid = torch.zeros(batch_size)
    eid = torch.zeros(batch_size)
    attr = torch.stack(
        [depature, total_dis, total_time, total_len, avg_dis, avg_speed, sid, eid], dim=1)

    # Create Model
    unet = Guide_UNet(config)

    # Sample input
    x = torch.randn(batch_size, features, seq_len)

    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f'{total_params:,} total parameters.')

    # Create a sample output
    out = unet(x, t, attr)
    print(out.shape)
