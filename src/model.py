import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwarmAttentionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, embeddings, adjacency):
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)

        scores = torch.matmul(Q, K.t()) / math.sqrt(self.embed_dim)
        mask = (adjacency == 0)
        scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        return self.out_proj(context)

class HardenedLegionNet(nn.Module):
    def __init__(self, c_in=3, n_actions=5, grid_size=40):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        convh = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        flat_size = convw * convh * 32

        self.fc_vis = nn.Linear(flat_size, 64)
        self.class_embed = nn.Embedding(2, 16)
        self.attention = SwarmAttentionHead(embed_dim=80)
        self.fc_q = nn.Linear(160, n_actions)

        self.register_buffer('rho', torch.tensor(0.0))

    def forward(self, img_batch, adjacency, class_ids):
        x = F.relu(self.conv1(img_batch))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        vis_feat = F.relu(self.fc_vis(x))
        cls_feat = self.class_embed(class_ids)
        my_embedding = torch.cat([vis_feat, cls_feat], dim=1)
        swarm_context = self.attention(my_embedding, adjacency)
        combined = torch.cat([my_embedding, swarm_context], dim=1)
        return self.fc_q(combined)

class AeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128); self.fc2 = nn.Linear(128, 9)
        self.register_buffer('rho', torch.tensor(0.0))
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TraderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 5); self.fc = nn.Linear(16*36, 3)
        self.register_buffer('rho', torch.tensor(0.0))
    def forward(self, x):
        return self.fc(F.relu(self.conv(x)).view(x.size(0), -1))
