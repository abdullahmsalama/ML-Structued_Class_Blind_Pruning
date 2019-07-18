import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weight_prune_entireneuron_classblind_AlexNet(net, pruning_perc):
    all_weights = []
    threshold=np.zeros(1)
    for name, p in net.named_parameters():
        if "weight" in name:
            if len(p.data.size()) == 4:
                all_weights += list((p.cpu().data.abs().sum(1).sum(1).sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
            elif len(p.data.size()) == 2 and p.data.size(0) != 1000:
                all_weights += list((p.cpu().data.abs().sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
    threshold = np.percentile(np.array(all_weights), pruning_perc)
    
    masks = []
    masks2 = []
    for n,p in net.named_parameters():
        if "weight" in n:
            if len(p.data.size()) == 4:
                temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold).float()
                pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold).float()
                pruned_inds = torch.ones(p.size()).cuda()
                for j in range(0,temp.size(0)):
                        pruned_inds[j,:,:,:] = temp[j]*torch.ones(p.size(1),p.size(2),p.size(3)).cuda()
                        
            elif len(p.data.size()) == 2 and p.data.size(0) != 1000:
                pruned_inds_bias = ((p.data.abs().sum(1))/(p.nelement()/(p.size(0))) > threshold).float()
                pruned_inds = ((((p.data.abs().sum(1))/(p.nelement()/(p.size(0))) > threshold).float())*((torch.ones(p.size()).cuda()).permute(1,0))).permute(1,0)
            elif len(p.data.size()) == 2 and p.data.size(0) == 1000:
                pruned_inds = torch.ones(p.size())
                pruned_inds_bias = torch.ones(p.size(0))   
            masks.append(pruned_inds.float())
            masks2.append(pruned_inds_bias.float())
    return masks, masks2

def weight_prune_entireneuron_classblind_ResNet50(net, pruning_perc, flag):
    all_weights = []
    threshold=np.zeros(1)
    for name, p in net.named_parameters():
        if "weight" in name:
            if "conv" in name:
                all_weights += list((p.cpu().data.abs().sum(1).sum(1).sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
            elif "fc1" in name:
                all_weights += list((p.cpu().data.abs().sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
            elif ("downsample" in name) and (len(p.data.size()) != 1) :
                all_weights += list((p.cpu().data.abs().sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
                
    threshold = np.percentile(np.array(all_weights), pruning_perc)
    
    masks = []
    names=[]
    count=0
    saved='ss'
    threshold_conv3=np.zeros(1)
    for n,p in net.named_parameters():
        if (len(p.data.size()) != 1) or ((len(p.data.size()) == 1) and  ("weight" in n)):
            names.append(n)
        if "weight" in n:
            s= n.split(".")
            if s[0]!=saved and "conv3" in n:
                count=0
        
            if ("conv3" in n) and (len(p.data.size()) == 4):
                count+=1
                s= n.split(".")
                if count==1:
                    saved=s[0]
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold).float()
                elif count==2:
                    weights=[]
                    weights = list((p.cpu().data.abs().sum(1).sum(1).sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
                    threshold_conv3 = np.percentile(np.array(weights),(1-((torch.nonzero(masks[len(masks)-8]).size(0))/(masks[len(masks)-8].nelement())))*100)
                    
                    
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    
                    if flag ==1:
                        temp = masks[len(masks)-7]
                        pruned_inds_bias = masks[len(masks)-7]
                else:
                    weights = []
                    weights = list((p.cpu().data.abs().sum(1).sum(1).sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
                    threshold_conv3 = np.percentile(np.array(weights), (1- ((torch.nonzero(masks[len(masks)-6]).size(0))/(masks[len(masks)-6].nelement())))*100)
                    
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    if flag ==1:
                        temp = masks[len(masks)-5]
                        pruned_inds_bias = masks[len(masks)-5]
                                 
                    
                pruned_inds = torch.ones(p.size()).cuda()
                for j in range(0,temp.size(0)):
                        pruned_inds[j,:,:,:] = temp[j]*torch.ones(p.data.size(1),p.data.size(2),p.data.size(3)).cuda()
                
                masks.append(pruned_inds.float()) 
                
            elif ("conv" in n) and (p.size(len(p.data.size())-1)!=1) and ("conv3" not in n):
                if p.size(0)==p.size(0):
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold).float()
                else:
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(torch.nonzero(p).size(0)/(p.size(0))) > 0).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(torch.nonzero(p).size(0)/(p.size(0))) > 0).float()
                    
                pruned_inds = torch.ones(p.size()).cuda()
                for j in range(0,temp.size(0)):
                        pruned_inds[j,:,:,:] = temp[j]*torch.ones(p.size(1),p.size(2),p.size(3)).cuda()
                           
                masks.append(pruned_inds.float())
                
            elif ('bn' in n) or (len(p.data.size()) == 1):
                masks.append(pruned_inds_bias.float())
                if "bias" in n:

            elif "linear" in n :
                pruned_inds = torch.ones(p.size()).cuda()
                pruned_inds_bias = torch.ones(p.size(0))

                masks.append(pruned_inds.float())
            
            elif "fc1" in n:
                pruned_inds_bias = ((p.data.abs().sum(1))/(p.nelement()/(p.size(0))) > threshold).float()
                pruned_inds = ((((p.data.abs().sum(1))/(p.nelement()/(p.size(0))) > threshold).float())*((torch.ones(p.size()).cuda()).permute(1,0))).permute(1,0)
                
                masks.append(pruned_inds.float())
            
            elif (("downsample" in n) and (len(p.data.size()) != 1)) or (p.size(len(p.data.size())-1)==1 and p.size(len(p.data.size())-2)==1) :
                
                if "downsample" in n:
                    weights=[]
                    weights = list((p.cpu().data.abs().sum(1).sum(1).sum(1).numpy().flatten())/(p.nelement()/(p.size(0))))
                    threshold_conv3 = np.percentile(np.array(weights),(1-((torch.nonzero(masks[len(masks)-1]).size(0))/(masks[len(masks)-1].nelement())))*100)
                    
                    temp= (p.data.abs().sum(1).sum(1).sum(1)/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).sum(1).sum(1))/(p.nelement()/(p.size(0))) > threshold_conv3).float()
                    
                    if flag ==1:
                        temp = masks[len(masks)-1]
                        pruned_inds_bias = masks[len(masks)-1]
                else:
                    temp= (p.data.abs().sum(1).view(-1)/(p.data.nelement()/(p.data.size(0))) > threshold).float()
                    pruned_inds_bias= ((p.data.abs().sum(1).view(-1))/(p.data.nelement()/(p.data.size(0))) > threshold).float()
                  
                pruned_inds = torch.ones(p.size()).cuda()
                for j in range(0,temp.size(0)):
                        pruned_inds[j,:,:,:] = temp[j]*torch.ones(p.data.size(1),p.data.size(2),p.data.size(3)).cuda()
                 
                masks.append(pruned_inds.float()) 
                
            elif "fc" in n :
                pruned_inds = torch.ones(p.size()).cuda()
                pruned_inds_bias = torch.ones(p.size(0))
   
                masks.append(pruned_inds.float())         
                
                
    return masks, names