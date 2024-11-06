import re
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import torch 

class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class KLLossSoft(torch.nn.modules.loss._Loss):
    def forward(self, output, target, T=1.0):
        output = output[0] if isinstance(output, tuple) else output
        target = target[0] if isinstance(target, tuple) else target
        output, target = output / T, target / T
        target_prob = F.softmax(target, dim=1)
        output_log_prob = F.log_softmax(output, dim=1)
        loss = - torch.sum(target_prob * output_log_prob, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

 
class ChannelWiseDivergence(nn.Module):
    def __init__(self, tau=1.0, loss_weight=1.0, size_average=True):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.size_average = size_average

    def forward(self, preds_S, preds_T):
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        if self.size_average:
            loss = self.loss_weight * loss.mean()
        else:
            loss = self.loss_weight * loss.sum()
        return loss
   
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    def forward(self,featmap):
        n,c,h,w = featmap.shape
        featmap = featmap.reshape((n,c,-1))
        featmap = featmap.softmax(dim=-1)
        return featmap
 
class CriterionCWD(nn.Module):
    
    def __init__(self,norm_type='channel',divergence='kl',temperature=4.0, loss_weight=3.0):
    
        super(CriterionCWD, self).__init__()
       

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type =='spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x:x.view(x.size(0),x.size(1),-1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type
        self.loss_weight = loss_weight
        self.temperature = temperature

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

        
        

    def forward(self,preds_S, preds_T):
        
        n,c,h,w = preds_S.shape
        #import pdb;pdb.set_trace()
        if self.normalize is not None:
            norm_s = self.normalize(preds_S/self.temperature)
            norm_t = self.normalize(preds_T.detach()/self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()
        
        
        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s,norm_t)
        
        #item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        #import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
            # loss /= n * h * w
        else:
            loss /= n * h * w

        return loss * (self.temperature**2) * self.loss_weight


class MultiClassDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        # Convert `label` to one-hot encoding, add an extra class for label 255
         # Replace 255 with `pred.shape[1]`
        # print(torch.unique(label))
        label = torch.where(label == 255, torch.tensor(pred.shape[1], dtype=label.dtype, device=label.device), label)

        label_one_hot = F.one_hot(label, num_classes=pred.shape[1] + 1).permute(0, 3, 1, 2).float()
        
        # Ignore the extra class in `label_one_hot` for Dice loss computation
        label_one_hot = label_one_hot[:, :-1, :, :]
        
        # Apply softmax to `pred`
        pred = F.softmax(pred, dim=1)

        pred = pred.contiguous()
        label_one_hot = label_one_hot.contiguous()

        # Set `pred` for pixels where `label` is 255 to 0
        # valid_mask = (label != 255)
        # pred = pred * valid_mask.unsqueeze(1)
        
        intersection = (pred * label_one_hot).sum(dim=(2, 3))
        cardinality = pred.sum(dim=(2, 3)) + label_one_hot.sum(dim=(2, 3))

        dice_loss = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_loss.mean(dim=0)
        
        return dice_loss.mean()






