import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """
    一个支持RoPE的交叉注意力Transformer Block。
    """
    def __init__(self, dim, num_heads, rope, mlp_ratio=4.0, dropout_p=0.0):
        """
        初始化模块。

        参数:
        dim (int): 输入Token的特征维度 (768)。
        num_heads (int): 多头注意力的头数 (12)。
        rope (nn.Module): 您已经实例化的RoPE模块。
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        # --- 第一部分: 交叉注意力 (手动实现) ---
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # 定义Q, K, V的线性投射层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # 定义输出投射层
        self.o_proj = nn.Linear(dim, dim)
        
        # --- 第二部分: 前馈网络 (MLP) ---
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, query, kv, pos_q, pos_kv):
        """
        前向传播。

        参数:
        query (torch.Tensor): 查询序列, 形状 (B, P_q, C)。
        kv (torch.Tensor): 键/值序列, 形状 (B, P_kv, C)。
        pos_q (torch.Tensor): query对应的位置编码信息。
        pos_kv (torch.Tensor): kv对应的位置编码信息。
        """
        B, P_q, C = query.shape
        _, P_kv, _ = kv.shape

        # --- 交叉注意力部分 ---
        query_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)

        # 1. 线性投射得到 Q, K, V
        q = self.q_proj(query_norm) # (B, P_q, C)
        k = self.k_proj(kv_norm)   # (B, P_kv, C)
        v = self.v_proj(kv_norm)   # (B, P_kv, C)

        # 2. 将Q, K, V分割成多头
        # (B, SeqLen, Dim) -> (B, SeqLen, num_heads, head_dim) -> (B, num_heads, SeqLen, head_dim)
        q = q.view(B, P_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, P_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, P_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 应用RoPE！
        if self.rope is not None:
            q = self.rope(q, pos_q)
            k = self.rope(k, pos_kv)

        # 4. 计算注意力分数 (Scaled Dot-Product)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 5. 用注意力分数加权V
        attn_output = attn_probs @ v # (B, num_heads, P_q, head_dim)

        # 6. 合并多头并进行输出投射
        # (B, num_heads, P_q, head_dim) -> (B, P_q, num_heads, head_dim) -> (B, P_q, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, P_q, C)
        attn_output = self.o_proj(attn_output)
        
        # 第一个残差连接
        query = query + attn_output
        
        # --- MLP部分 ---
        mlp_output = self.mlp(self.norm_mlp(query))
        query = query + mlp_output
        
        return query
    
class SelfAttentionBlockWithRoPE(nn.Module):
    def __init__(self, dim, num_heads, rope, mlp_ratio=4.0, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = rope

        self.norm1 = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3) # 一次性投射 Q, K, V
        self.o_proj = nn.Linear(dim, dim)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_features, dim),
            nn.Dropout(dropout_p)
        )

    def forward(self, x, pos):
        B, SeqLen, C = x.shape
        x_norm = self.norm1(x)

        # 1. 线性投射得到 Q, K, V
        qkv = self.qkv_proj(x_norm).reshape(B, SeqLen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # 形状: (B, num_heads, SeqLen, head_dim)

        # 2. 应用RoPE
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # 3. 计算注意力
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, SeqLen, C)
        attn_output = self.o_proj(attn_output)
        
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class PairwiseInteractionModelWithRoPE(nn.Module):
    def __init__(self, dim=768, num_blocks=4, num_heads=12, head_dim=64, rope=None):
        super().__init__()
        self.dim = dim

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.segment_embedding = nn.Embedding(2, dim)
        
        # 实例化RoPE模块
        self.rope = rope

        # 使用我们自定义的Block来构建Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlockWithRoPE(dim=dim, num_heads=num_heads, rope=self.rope)
            for _ in range(num_blocks)
        ])

    def forward(self, query_tokens, ref_tokens, pos_q, pos_r):
        """
        前向传播。

        参数:
        query_tokens (torch.Tensor): 查询图Token, 形状 (B, P_q, C)。
        ref_tokens (torch.Tensor): 参考图Token, 形状 (B, N_ref, P_r, C)。
        pos_q (torch.Tensor): 查询图位置编码, 形状 (B, P_q, 2)。
        pos_r (torch.Tensor): 参考图位置编码, 形状 (B, N_ref, P_r, 2)。
        """
        B, N_ref, P_r, C = ref_tokens.shape
        P_q = query_tokens.shape[1]

        # --- 1. 准备并行的批次数据 (包括位置编码) ---
        query_tokens_exp = query_tokens.unsqueeze(1).expand(-1, N_ref, -1, -1)
        pos_q_exp = pos_q.unsqueeze(1).expand(-1, N_ref, -1, -1)
        
        query_flat = query_tokens_exp.reshape(B * N_ref, P_q, C)
        ref_flat = ref_tokens.reshape(B * N_ref, P_r, C)
        pos_q_flat = pos_q_exp.reshape(B * N_ref, P_q, 2)
        pos_r_flat = pos_r.reshape(B * N_ref, P_r, 2)

        # --- 2. 准备输入序列 (包括位置编码) ---
        # 特征
        full_sequence = torch.cat([query_flat, ref_flat], dim=1)
        
        # 位置
        full_pos = torch.cat([pos_q_flat, pos_r_flat], dim=1)
        
        # --- 3. 依次通过自定义的Transformer Block ---
        for block in self.transformer_blocks:
            full_sequence = block(full_sequence, full_pos)
            
        # --- 4. 提取被“丰富”了的查询Token ---
        enriched_query_tokens_flat = full_sequence[:, 1:1+P_q, :]
        
        # --- 5. 将结果恢复成 (B, N_ref, P, C) 的形状 ---
        enriched_query_tokens = enriched_query_tokens_flat.view(B, N_ref, P_q, C)
        
        return enriched_query_tokens