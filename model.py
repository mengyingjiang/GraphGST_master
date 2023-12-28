from layer import *
import os

class GraphGST(nn.Module):
    def __init__(self, num_classes, bonds, nlayers, heads, num_position, emb_dropout, r_positive):
        super().__init__()
        bonds = bonds
        self.bn1 = torch.nn.BatchNorm1d(bonds)
        self.posi_encoding = nn.Linear(num_position, bonds)
        self.layer0 = EncoderLayer(bonds, heads)
        self.layers = torch.nn.ModuleList([EncoderLayer(bonds, heads) for _ in range(nlayers-1)])
        self.mapp = nn.Linear(num_position, bonds)
        self.position_encoder = position_Encoder(num_position, bonds, heads)
        self.disc = Discriminator(bonds)
        self.dropout = nn.Dropout(emb_dropout)
        self.r_positive = r_positive
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(bonds*(nlayers+2)), #nlayers+2
            nn.Linear(bonds*(nlayers+2), num_classes)
        )
        self.ac = gelu

    def forward(self, x, x_position_original):

        x_cat = x_embed = self.layer0(x)
        x_cat = self.dropout(x_cat)
        for layer in self.layers:
            x_embed = layer(x_embed)
            x_cat = torch.cat((x_cat,x_embed),dim=1)
        x_position_embed = self.position_encoder(x_position_original, x_embed)
        x_cat = torch.cat((x, x_cat, x_position_embed), dim=1)
        ret_os = self.disc(x_embed, x_position_embed, x_position_original, self.r_positive)

        return self.mlp_head(x_cat), ret_os
