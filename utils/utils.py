import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import torch.nn.functional as F
import copy

def to_var(x, requires_grad=False, volatile=False):
    """
    Variable type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def copy_weights(model_from, model_to):
    x=0
    y=0
    filters_for_next=0
    filters_for_next_downsample=[]
    for n,p in model_from.named_parameters():
        x=x+1
        y=0
        for name,param in model_to.named_parameters():
            y=y+1
            if x==y:
                if (n!=name):
                    assert(n==name)
                    
                    
                if len(p.data.size()) == 4 and n == "conv1.weight":
                    filters_before=[0,1,2]
                    nonzero=torch.nonzero(p.data.sum(1).sum(1).sum(1))
                    for j in range(0,param.size(0)):
                        for i in range (0,len(filters_before)):
                            param[j,i,:,:]=p[nonzero[j].item(),filters_before[i],:,:]
                            
                    filters_for_next_conv1 = nonzero
                elif (len(p.data.size()) == 4) and (n != "conv1.weight") and ("downsample" not in n):
                    nonzero=torch.nonzero(p.data.sum(1).sum(1).sum(1))
                    for j in range(0,param.size(0)):
                        for i in range (0,len(filters_before)):
                            param[j,i,:,:]=p[nonzero[j].item(),filters_before[i].item(),:,:]
                    if "conv3" in n:
                        filters_for_next_downsample.append(nonzero)                  
                            
                elif ("downsample" in n) and ("layer1.0.downsample" in n) and (len(p.data.size()) == 4) :
                    nonzero=torch.nonzero(p.data.sum(1).sum(1).sum(1))
                    for j in range(0,param.size(0)):
                        for i in range (0,len(filters_for_next_conv1)):
                            param[j,i,:,:]=p[nonzero[j].item(),filters_for_next_conv1[i].item(),:,:]                    

                elif ("downsample" in n) and ("layer1.0.downsample" not in n) and (len(p.data.size()) == 4):
                    nonzero=torch.nonzero(p.data.sum(1).sum(1).sum(1))
                    filters_before=filters_for_next_downsample[len(filters_for_next_downsample)-2]
                    for j in range(0,param.size(0)):
                        for i in range (0,len(filters_before)):
                            param[j,i,:,:]=p[nonzero[j].item(),filters_before[i].item(),:,:]                              
                            
                elif len(p.data.size()) == 2:
                    nonzero=torch.nonzero(p.data.sum(1))
                    for j in range(0,param.size(0)):
                        for i in range (0,len(filters_before)):
                            param[j,i]=p[nonzero[j].item(),filters_before[i].item()]
                elif len(p.data.size()) == 1:
                    nonzero=torch.nonzero(p.data)
                    filters_before=nonzero
                    for j in range(0,param.size(0)):
                        param[j]=p[nonzero[j].item()]
                        
                        if (len(p.data.size()) == 1 and "bias" not in n) :
                            if len(n.split("."))>2 and len(n.split("."))<5:
                                string_name= n.split(".")
                                sub_string = str('{}[{}].{}[{}]'.format(string_name[0],string_name[1],string_name[2],string_name[3]))
                                
                                
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_mean',
                                                                                                 'j', eval('sub_string'),
                                                                                                 'running_mean',
                                                                                                 'nonzero[j].item()')))
                                
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_var',
                                                                                                 'j', eval('sub_String'),
                                                                                                 'running_var',
                                                                                                 'nonzero[j].item()')))
                                

                            elif len(n.split("."))>=5:
                                string_name= n.split(".")
                                sub_string = str('{}[{}].{}[{}]'.format(string_name[0],string_name[1],string_name[2],string_name[3]))
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_mean',
                                                                                                 'j', eval('sub_String'),
                                                                                                 'running_mean',
                                                                                                 'nonzero[j].item()'))) 
                                
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_var',
                                                                                                 'j', eval('sub_string'),
                                                                                                 'running_var',
                                                                                                 'nonzero[j].item()'))) 
                            
                            else:
                                string_name= n.split(".")
                                sub_string = str('{}'.format(string_name[0]))                   
                    
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_mean',
                                                                                                 'j', eval('sub_string'),
                                                                                                 'running_mean',
                                                                                                 'nonzero[j].item()'))) 
                                eval(str('model_to.{}.{}[{}].copy_(model_from.{}.{}[{}])'.format(eval('sub_string'), 'running_var',
                                                                                                 'j', eval('sub_string'),
                                                                                                 'running_var',
                                                                                                 'nonzero[j].item()')))                        
                break
    return model_to

def get_dimensions(model):
    layers=[]
    dimensions=[]
    i=0
    for n,p in model.named_parameters():
        if "conv" in n:
            if "downsample" not in n:
                i+=1
                dimensions.append(torch.nonzero((p).sum(1).sum(1).sum(1)).size(0))
                layers.append(i)
    return dimensions