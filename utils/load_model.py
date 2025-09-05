import torch
import copy
import gc

from vggt.models.vggt import VGGT, Aggregator
from src.models.pose_head import PoseHead
from src.models.pose_head_split import PoseHeadSplit
from src.models.bbox_conf import BBoxConf
from src.models.pggt import PGGT
from src.models.pixel_voting_head import VectorMapDecoder


def load_model(config, device="cuda"):
    print("Loading model...")
    config_model = config.model

    if config_model.backbone.name == "vggt" and config_model.backbone.pretrained_vggt is not None:
        vggt_model = VGGT.from_pretrained(config_model.backbone.pretrained_vggt)
        backbone = copy.deepcopy(vggt_model.aggregator)
        del vggt_model # Free memory
    elif config_model.backbone.name == "vggt":
        backbone = Aggregator(
            img_size=config_model.backbone.img_size,
            patch_size=config_model.backbone.patch_size,
            embed_dim=config_model.backbone.embed_dim,
            depth=config_model.backbone.depth,
            num_heads=config_model.backbone.num_heads,
            mlp_ratio=config_model.backbone.mlp_ratio,
            patch_embed=config_model.backbone.patch_embed,
            load_pretrained_dino_weights=config_model.backbone.load_pretrained_dino_weights
        )
    
    pose_head = VectorMapDecoder()
    # if config_model.posehead.name == "posehead":
    #     pose_head = PoseHead(config_model.posehead)
    # elif config_model.posehead.name == "bboxconf":
    #     pose_head = BBoxConf(config_model.posehead)
    

    model = PGGT(config_model, backbone, pose_head).to(device)       

    if config.checkpoint is not None:
        print(f"Loading checkpoint form {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model