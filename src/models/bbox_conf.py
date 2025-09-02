import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.heads.head_act import activate_pose

class BBoxConf(nn.Module):
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

        # if self.pose_encoding_type == "SITE_6DRot":
        #     self.target_dim = 9
        #     self.rot_dim = 6
        #     self.trans_dim = 3
        # else:
        #     raise ValueError(f"Unsupported object encoding type: {self.pose_encoding_type}")
        self.target_dim = 16
        
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
        self.empty_bbox_token = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_bbox = nn.Linear(self.target_dim, self.dim_in)  # Linear layer to embed pose token into input dimension

        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.dim_in, 3 * self.dim_in, bias=True))

        # Adaptive layer normalization without affine parameters.
        self.adaln_norm = nn.LayerNorm(self.dim_in, elementwise_affine=False, eps=1e-6)
        # self.pose_branch = Mlp(
        #     in_features=self.dim_in,
        #     hidden_features=self.dim_in // 2,
        #     out_features=self.target_dim,
        #     drop=0,
        # )
        # 创建独立的旋转和平移分支
        self.bbox_branch = Mlp(
            in_features=self.dim_in,
            hidden_features=self.dim_in // 2,
            out_features=self.target_dim, # 输出维度为旋转部分
            drop=0.1,
        )
        # self.confidence_branch = Mlp(
        #     in_features=self.dim_in,
        #     hidden_features=self.dim_in // 2,
        #     out_features=1,  # 每个姿态对应1个平移和旋转置信度分数
        #     drop=0.1,
        # )

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
        conf_tokens = tokens[:, :, 1]
        # conf_tokens = conf_tokens + conf_tokens[:, :1, :]
        conf_tokens = self.token_norm(conf_tokens)

        pred_pose_enc_list = self.trunk_fn(pose_tokens, conf_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, bbox_tokens: torch.Tensor, conf_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        Iteratively refine pose predictions.

        Args:
            pose_tokens (torch.Tensor): Normalized pose tokens with shape [B, 1, C].
            num_iterations (int): Number of refinement iterations.

        Returns:
            list: List of activated pose encodings from each iteration.
        """
        B, S, C = bbox_tokens.shape  # S is expected to be 1.
        pred_bbox_enc = None
        output_list = []
        pred_conf_list = []
        pred_bbox_enc_list = []

        for _ in range(num_iterations):
            # Use a learned empty pose for the first iteration.
            if pred_bbox_enc is None:
                module_input = self.embed_bbox(self.empty_bbox_token.expand(B, S, -1))
            else:
                # Detach the previous prediction to avoid backprop through time.
                pred_bbox_enc = pred_bbox_enc.detach()
                module_input = self.embed_bbox(pred_bbox_enc)

            # Generate modulation parameters and split them into shift, scale, and gate components.
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # Adaptive layer normalization and modulation.
            bbox_tokens_modulated = gate_msa * modulate(self.adaln_norm(bbox_tokens), shift_msa, scale_msa)
            bbox_tokens_modulated = bbox_tokens_modulated + bbox_tokens

            # Pass through the trunk.
            trunk_out = self.trunk(bbox_tokens_modulated)
            trunk_out = self.trunk_norm(trunk_out)

            # # Predict pose.
            # pred_pose_enc_delta = self.pose_branch(trunk_out)

            # if pred_pose_enc is None:
            #     pred_pose_enc = pred_pose_enc_delta
            # else:
            #     pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            # --- 修改开始 ---
            # 预测姿态delta
            pred_bbox_enc_delta = self.bbox_branch(trunk_out)

            # 预测置信度 logit (未经激活)
            # pred_confidence_logit = self.confidence_branch(conf_tokens)
            Q = conf_tokens[:, :1, :]
            F = conf_tokens
            Q_norm = Q / Q.norm(dim=-1, keepdim=True)
            F_norm = F / F.norm(dim=-1, keepdim=True)
            Q_expand = Q_norm.expand(-1, F.size(1), -1)  # (B, N, C)
            pred_confidence_logit = (Q_expand * F_norm).sum(dim=-1)[..., None]  # (B, N)
            # --- 修改结束 ---

            if pred_bbox_enc is None:
                pred_bbox_enc = pred_bbox_enc_delta
            else:
                pred_bbox_enc = pred_bbox_enc + pred_bbox_enc_delta
            
            activated_bbox = pred_bbox_enc
            # 对预测进行激活
            # activated_bbox = activate_pose(
            #     pred_bbox_enc,
            #     trans_act=self.trans_act,
            #     quat_act=self.quat_act,
            # )
            
            # 将每次迭代的结果（姿态）都存起来
            pred_bbox_enc_list.append(activated_bbox)
            # 将每次迭代的结果（置信度）都存起来
            pred_conf_list.append(pred_confidence_logit)
            # # 将每次迭代的结果（姿态和置信度）都存起来
            # output_list.append((activated_pose, pred_confidence_logit))

        return pred_bbox_enc_list, pred_conf_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift