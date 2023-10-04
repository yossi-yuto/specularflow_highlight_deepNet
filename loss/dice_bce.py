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
        # self.BCE_loss = nn.BCELoss()
        
    def forward(self, pred, target, target_edge) -> tuple: 
        num_map = len(pred)
        spec_map_loss = 0
        for i in range(num_map):
            if i < num_map - 2:
                spec_map_loss+= self.W_s * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
            if i == (num_map - 2):
                edge_loss = self.W_b * (self.BCEwithLogitsLoss(pred[i], target_edge))
            elif i == (num_map - 1):
                final_loss = self.W_f * (self.BCEwithLogitsLoss(pred[i], target) + self.dice_loss(pred[i], target))
                
        return spec_map_loss, edge_loss, final_loss 
        


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
        assert len(pred) == 2, f"The number of predict maps is not 2 (boudary map, final map)."
        sum_loss = 0
        sum_loss += self.W_map * self.BCEwithLogitsLoss(pred[0], target_edge)
        sum_loss += self.W_final * (self.BCEwithLogitsLoss(pred[1], target) + self.dice_loss(pred[1], target))
        return sum_loss