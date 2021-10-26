from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cosine(nn.Module):
    
    def __init__(self):
        super(Cosine, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        
        bsz = f_s.shape[0] #64*
        f_s = f_s.view(bsz, -1)#64*dim
        f_s = torch.nn.functional.normalize(f_s)#64*dim
        
        f_t = f_t.view(bsz, -1)
        f_t = torch.nn.functional.normalize(f_t)#64*dim

        G_s = torch.mm(f_s, torch.t(f_s))#64*dim
        # G_s = G_s / G_s.norm(2)
        
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        
        G_diff = G_t - G_s
        
        #print('G_diff0', G_diff)
        
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        
        return loss
