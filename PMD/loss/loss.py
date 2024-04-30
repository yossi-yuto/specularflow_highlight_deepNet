import pdb
import torch.nn as nn

from loss.dice import DiceLoss

# componet lossの計算時
class LossComponent(nn.Module):
    def __init__(self, W_map=1, W_final=1):
        super(LossComponent, self).__init__()
        self.W_map = W_map
        self.W_final = W_final
        
        self.BCEwithLogitsLoss =  nn.BCEWithLogitsLoss() 
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target, target_edge) -> tuple:
        assert len(pred) == 5, f"The number of predict maps is not 5."
        sum_loss = 0
        for i in range(len(pred)):
            sum_loss += self.W_map * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
        return sum_loss
            
            
# componet lossの計算時
class Loss_Refine_EDF(nn.Module):
    def __init__(self, W_edge=5, W_final=1):
        super(Loss_Refine_EDF, self).__init__()
        self.W_map = W_edge
        self.W_final = W_final
        
        self.BCEwithLogitsLoss =  nn.BCEWithLogitsLoss() 
        self.dice_loss = DiceLoss()
        
    def forward(self, pred, target, target_edge) -> tuple:
        sum_loss = 0
        sum_loss += self.W_map * self.BCEwithLogitsLoss(pred[-2], target_edge)
        sum_loss += self.W_final * (self.BCEwithLogitsLoss(pred[-1], target) + self.dice_loss(pred[-1], target))
        return sum_loss