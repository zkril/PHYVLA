"""Variable-size physical token set and visual-query residual fusion."""
import torch
from torch import nn

class PhysicsTokenizerV2(nn.Module):
    def __init__(self, robot_dim, object_dim, width=128):
        super().__init__()
        self.config = dict(robot_dim=robot_dim, object_dim=object_dim, width=width)
        self.robot_encoder = nn.Sequential(nn.Linear(robot_dim,width),nn.GELU(),nn.Linear(width,width))
        self.object_encoder = nn.Sequential(nn.Linear(object_dim,width),nn.GELU(),nn.Linear(width,width))
        self.types = nn.Parameter(torch.zeros(2,width))
        layer = nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True)
        self.context = nn.TransformerEncoder(layer,1)
        self.robot_decoder = nn.Linear(width,robot_dim)
        self.object_decoder = nn.Linear(width,object_dim)

    def forward(self, robot, objects, mask):
        z = torch.cat([self.robot_encoder(robot)[:,None]+self.types[0],
                       self.object_encoder(objects)+self.types[1]],1)
        valid = torch.cat([torch.ones_like(mask[:,:1]),mask],1).bool()
        return self.context(z,src_key_padding_mask=~valid), valid

    def loss(self, robot, objects, mask):
        z, valid = self(robot,objects,mask)
        r = (self.robot_decoder(z[:,0])-robot).square().mean()
        o = (self.object_decoder(z[:,1:])-objects).square().mean(-1)
        return r+(o*mask).sum()/mask.sum(), {"robot_mse":r.detach(),
            "object_mse":((o*mask).sum()/mask.sum()).detach()}

class PhysicsResidualFusion(nn.Module):
    def __init__(self, hidden_dim=2560, width=128):
        super().__init__()
        self.config = dict(hidden_dim=hidden_dim,width=width)
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,width))
        self.attention = nn.MultiheadAttention(width,4,dropout=0,batch_first=True)
        self.output = nn.Linear(width,hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, queries, tokens, valid, scale=1.0):
        context,_ = self.attention(self.query(queries),tokens,tokens,
                                   key_padding_mask=~valid,need_weights=False)
        return queries + scale*self.output(context)
