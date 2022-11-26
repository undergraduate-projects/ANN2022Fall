"""
:Author :GodofTheFallen
:Time:  :2022/11/13
:File   :functions.py
:content:Criss-Cross Attention Module.

:Reference: git@github.com:speedinghzl/CCNet
"""

import jittor as jt
from jittor import nn


def INF(B, H, W):
    return -jt.diag(jt.Var(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module """
    
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self._INF = INF
        self.gamma = jt.Var(jt.zeros(1)).start_grad()
    
    def execute(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).view(m_batchsize * height, -1, width)
        energy_H = (jt.bmm(proj_query_H, proj_key_H) +
                    self._INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = jt.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(jt.concat([energy_H, energy_W], 3))
        
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].view(m_batchsize * height, width, width)
        out_H = jt.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = jt.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


# if __name__ == '__main__':
#     jt.flags.use_cuda = 1
#     model = CrissCrossAttention(64)
#     x = jt.randn(2, 64, 5, 6)
#     out = model(x)
#     print(out.shape)
