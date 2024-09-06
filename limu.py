import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, hidden, hidden_ff):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))

class LIMUBertEncoder(nn.Module):
    
    def __init__(self, d_model=120, feature_num=6, n_layers=4, hidden=72, hidden_ff=144, n_heads=4):
        super().__init__()
        
        # factorized embedding
        self.linear = nn.Linear(feature_num, hidden)
        self.pos_embed = nn.Embedding(d_model, hidden) # position embedding
        self.norm = nn.LayerNorm(hidden, eps=1e-12)
        
        self.n_layers = n_layers
        self.attn = nn.MultiheadAttention(hidden, n_heads)
        self.proj = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden, eps=1e-12)
        self.pwff = PositionWiseFeedForward(hidden, hidden_ff)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-12)
        
    def forward(self, x):
        # embedding
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len) # (S,) -> (B, S)

        # factorized embedding
        h = self.linear(x)
        h = h + self.pos_embed(pos)
        
        h = self.norm(h)
        
        for _ in range(self.n_layers):
            h = self.attn(h, h, h)[0]
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h
    

class LIMUBertModel(nn.Module):

    def __init__(self, seq_len=120, feature_num=6, n_layers=4, hidden=72, hidden_ff=144, n_heads=4, output_embed=False):
        super().__init__()
        self.encoder = LIMUBertEncoder(seq_len, feature_num, n_layers, hidden, hidden_ff, n_heads) # encoder
        self.fc = nn.Linear(hidden, hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.activ = F.gelu
        self.norm = nn.LayerNorm(hidden, eps=1e-12)
        self.decoder = nn.Linear(hidden, feature_num)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.encoder(input_seqs)
        if self.output_embed:
            return h_masked # only output this
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm
      
      
imu_input = torch.Tensor(torch.rand(10, 120, 6))

mdl = LIMUBertEncoder()
output = mdl.forward(imu_input)
print(output.shape) 