import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import to_var


class MaskedLinear_b(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear_b, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask,mask_bias):
        self.mask = to_var(mask, requires_grad=False)
        self.mask_bias = to_var(mask_bias, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data= self.bias.data*self.mask_bias.data
        self.mask_flag = True

    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            bias = self.bias*self.mask_bias
            if self.mask3_flag == True:
                weight = weight*self.mask3
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
        
class MaskedConv2d_b(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d_b, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
    
    def set_mask(self, mask,mask_bias):
        self.mask = to_var(mask, requires_grad=False)
        self.mask_bias = to_var(mask_bias, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data= self.bias.data*self.mask_bias.data
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            bias = self.bias*self.mask_bias
            return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        
        
        
class MaskedLinear_br(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear_br, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        #self.mask_bias = to_var(mask_bias, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        #self.bias.data= self.bias.data*self.mask_bias.data
        self.mask_flag = True
      
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            #bias = self.bias*self.mask_bias
            if self.mask3_flag == True:
                weight = weight*self.mask3
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
        
class MaskedConv2d_br(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d_br, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        #print("conv begin")
    
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        #self.mask_bias = to_var(mask_bias, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        #self.bias.data= self.bias.data*self.mask_bias.data
        self.mask_flag = True
    
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    
    
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            #bias = self.bias*self.mask_bias
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
class MaskedBatchNorm_br(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MaskedBatchNorm_br, self).__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 
        
        self.mask_flag = False
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data = self.bias.data*self.mask.data
        self.mask_flag = True
            
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            bias = self.bias*self.mask
            
            
            return F.batch_norm(x, self.running_mean, self.running_var, weight, bias,
                                  self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training or not self.track_running_stats, self.momentum, self.eps)
        
class MaskedBatchNorm1d_br(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d_br, self).__init__(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 
        
        self.mask_flag = False
        
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.bias.data = self.bias.data*self.mask.data
        self.mask_flag = True
            
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            bias = self.bias*self.mask
            return F.batch_norm(x, self.running_mean, self.running_var, weight, bias,
                                  self.training or not self.track_running_stats, self.momentum, self.eps)
        else:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  self.training or not self.track_running_stats, self.momentum, self.eps)
