import torch
import torch.nn as nn

from vggt.layers import Mlp

class VectorMapDecoder(nn.Module):
    def __init__(self, patch_size=14, in_dim=768, out_channels=16):
        super().__init__()
        
        # 目标是从 16x16 放大到 224x224，放大倍数是 14
        # 14 = 2 * 7，我们可以分两步上采样
        
        # 首先用一个卷积将高维特征映射到更易于处理的维度
        self.proj = nn.Conv2d(in_dim, 256, kernel_size=1)
        
        self.decoder = nn.Sequential(
            # 上采样 block 1: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            
            # 上采样 block 2: 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # 上采样 block 3: 64x64 -> 224x224 (stride=3.5, 不可行)
            # 这里需要一些技巧来处理非2的幂次方的上采样
            # 一种方法是使用 Upsample + Conv
            nn.Upsample(scale_factor=3.5, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 最终的预测头，将特征维度映射到我们想要的16
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

        # 置信度预测头
        self.confidence_branch = Mlp(
            in_features=in_dim,
            hidden_features=in_dim // 2,
            out_features=1,  # 每个姿态对应1个平移和旋转置信度分数
            drop=0.3,
        )

    def forward(self, x):
        # 预测置信度
        conf_tokens = x[:, :, 0, :]
        conf_tokens = conf_tokens + conf_tokens[:, :1, :]
        pred_conf = self.confidence_branch(conf_tokens)
        # Q = conf_tokens[:, :1, :]
        # F = conf_tokens
        # Q_norm = Q / Q.norm(dim=-1, keepdim=True)
        # F_norm = F / F.norm(dim=-1, keepdim=True)
        # Q_expand = Q_norm.expand(-1, F.size(1), -1)  # (B, N, C)
        # pred_conf = (Q_expand * F_norm).sum(dim=-1)[..., None]  # (B, N)

        # x 的输入形状: (b, n_maps, num_patches, in_dim)
        x = x[:, :, 5:, :] - x[:, :, 5:6, :] 
        b, n_maps, patch_count_s, in_dim = x.shape
        patch_count = torch.sqrt(torch.tensor(patch_count_s)).int()
        x = x.view(b * n_maps, -1, in_dim)
        # 1. 重塑成低分辨率空间特征图
        # (B*N, L, D) -> (B*N, H_p*W_p, D) -> (B*N, D, H_p, W_p)
        x = x.permute(0, 2, 1).contiguous() # -> (B*N, D, L)
        x = x.view(b * n_maps, -1,  patch_count, patch_count)
        # 此刻 x 的形状是 (34, 768, 16, 16)
        
        # 2. 通过解码器进行上采样
        x = self.proj(x)
        x = self.decoder(x)
        x = self.head(x)
        # 此刻 x 的形状是 (34, 16, 224, 224)
        
        # 3. 调整维度顺序以匹配 (H, W, C) 的格式
        x = x.permute(0, 2, 3, 1).contiguous() # -> (34, 224, 224, 16)
        
        # 4. 恢复 B 和 N_maps 维度
        x = x.view(b, n_maps, x.shape[1], x.shape[2], x.shape[3])
        # 最终形状: (2, 17, 224, 224, 16)

        return x, pred_conf