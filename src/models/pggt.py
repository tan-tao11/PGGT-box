import torch
import torch.nn as nn
import time

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
        pv_maps_input = data['pv_maps_input'].to(device)
        aggregated_tokens_list, patch_start_idx = self.backbone(images, pv_maps_input)
        with torch.autocast("cuda", enabled=False):
            pred_pv_map_offset, pred_conf = self.pose_head(aggregated_tokens_list[-1])

        return pred_pv_map_offset, pred_conf
    