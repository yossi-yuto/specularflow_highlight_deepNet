import pdb
import torch.nn as nn

from loss.dice import DiceLoss

class DBCE(nn.Module):
    def __init__(self,W_s = 1, W_b = 5, W_f = 2):
        super(DBCE, self).__init__()
        # weight
        self.W_s = W_s # 1 mirror maps
        self.W_b = W_b # 5  edge maps
        self.W_f = W_f # 2 final maps
        # loss obeject
        self.BCEwithLogitsLoss = nn.BCEWithLogitsLoss() 
        self.dice_loss = DiceLoss()
        self.BCE_loss = nn.BCELoss()
        
    def forward(self, pred, target, target_edge): 
        num_map = len(pred)
        sum_loss = 0
        for i in range(num_map):
            if i < num_map - 2:
                sum_loss += self.W_s * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
            if i == (num_map - 2):
                edge_loss = self.W_b * (self.BCEwithLogitsLoss(pred[i], target_edge))
                sum_loss += edge_loss
            
            elif i == (num_map - 1):
                final_loss = self.W_f * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
                sum_loss += final_loss
                
        return sum_loss, (final_loss + edge_loss)
        
