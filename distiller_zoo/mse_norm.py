from __future__ import print_function

import torch.nn as nn
import torch


class NORM_MSE(nn.Module):
    
    def __init__(self):
        super(NORM_MSE, self).__init__()
        self.MS = nn.MSELoss(reduction='none') #nn.MSELoss(size_average=False)

    def forward(self, output, target):
        
        target = target.view(target.shape[0], -1)
        
        output = output.view(output.shape[0], -1)
        
        magnitute = torch.norm(target,dim=1)
        
        magnitute_square = magnitute**2
        
        magnitute_square = torch.reshape(magnitute_square,(output.shape[0], -1) )
        
        
        loss = torch.sum( self.MS(output, target)/magnitute_square )/target.shape[0]
        
        return loss
