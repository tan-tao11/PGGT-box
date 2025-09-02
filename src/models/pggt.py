import torch
import torch.nn as nn


class PGGT(nn.Module):
    def __init__(self, config, backbone, pose_head):
        super().__init__()
        
        self.backbone = backbone
        self.pose_head = pose_head

        if config.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.config = config

    def forward(self, data, device):
        images = data['images'].to(device)
        bbox_2d_input = data['bbox_2d_input'].to(device)
        aggregated_tokens_list, patch_start_idx = self.backbone(images, bbox_2d_input)
        pred_pose_enc_list, pred_conf_list = self.pose_head(aggregated_tokens_list)

        if self.training:
            # If training, return whole output list
            return (pred_pose_enc_list, pred_conf_list)
        else:
            # If not training, return only the last output
            return ([pred_pose_enc_list[-1]], [pred_conf_list[-1]])
    