import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose

class PoseHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dim_in = config.dim_in
        self.trunk_depth = config.trunk_depth
        self.pose_encoding_type = config.pose_encoding_type
        self.num_heads = config.num_heads
        self.mlp_ratio = config.mlp_ratio
        self.init_values = config.init_values
        self.trans_act = config.trans_act
        self.quat_act = config.quat_act

        if self.pose_encoding_type == "SITE_6DRot":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported object encoding type: {self.pose_encoding_type}")
        
        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=self.dim_in,  # Input dimension of tokens 
                    num_heads=self.num_heads,  # Number of attention heads
                    mlp_ratio=self.mlp_ratio,  # Ratio of hidden dimension to input dimension
                    init_values=self.init_values,  # Initial value for the layer scale parameter
                )
                for _ in range(self.trunk_depth)  # Number of transformer blocks in the trunk
            ]
        )

        # Normalizations for pose tokens and trunk output.
        self.token_norm = nn.LayerNorm(self.dim_in)  # Normalization for pose tokens
        self.trunk_norm = nn.LayerNorm(self.dim_in)  # Normalization for trunk output

        # Learnable empty pose token.
        self.empty_pose_token = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, self.dim_in)  # Linear layer to embed pose token into input dimension

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.dim_in, 3 * self.dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(self.dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=self.dim_in,
            hidden_features=self.dim_in // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> list:
        """
        Forward pass to predict pose parameters.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            num_iterations (int, optional): Number of iterative refinement steps. Defaults to 4.

        Returns:
            list: A list of predicted pose encodings (post-activation) from each iteration.
        """
        # Use tokens from the last block for pose prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the pose tokens
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized pose tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated pose encodings from each iteration.
        """
        B, S, C = pose_tokens.shape  # S is expected to be 1.
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_token.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            # Pass through the trunk.
            trunk_out = self.trunk(pose_tokens_modulated)
            trunk_out = self.trunk_norm(trunk_out)

            # Predict pose.
            pred_pose_enc_delta = self.pose_branch(trunk_out)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # Apply final activation functions for translation and quaternion.
            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift