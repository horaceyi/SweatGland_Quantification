import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs=nn.Softmax(dim=1)(inputs)
        targets=F.one_hot(targets, num_classes=2).permute(0,3,1,2).contiguous()

        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        # inputs=targets
        inputs1=inputs[:,0,:,:]
        targets1=targets[:,0,:,:]
        intersection1 = (inputs1 * targets1).sum()                            
        dice1 = (2.*intersection1 + smooth)/(inputs1.sum() + targets1.sum() + smooth)  

        inputs2=inputs[:,1,:,:]
        targets2=targets[:,1,:,:]
        intersection2 = (inputs2* targets2).sum()                            
        dice2 = (2.*intersection2 + smooth)/(inputs2.sum() + targets2.sum() + smooth)  

        # inputs3=inputs[:,2,:,:]
        # targets3=targets[:,2,:,:]
        # intersection3 = (inputs3* targets3).sum()                            
        # dice3 = (2.*intersection3 + smooth)/(inputs3.sum() + targets3.sum() + smooth)  
        
        # dice=(dice1+dice2+dice3)/3.
        dice=dice1/2.+dice2/2.
        return 1 - dice