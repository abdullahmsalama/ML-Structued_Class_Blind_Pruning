import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torch.nn.init as init
from torch.autograd import Variable

from layer_types import MaskedConv2d_b, MaskedLinear_b

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            MaskedConv2d_b(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MaskedConv2d_b(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MaskedConv2d_b(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d_b(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv2d_b(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            MaskedLinear_b(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MaskedLinear_b(4096, 4096),
            nn.ReLU(inplace=True),
            MaskedLinear_b(4096, num_classes),
        )
    
    def set_masks(self,masks,masks2):
        self.features[0].set_mask(masks[0], masks2[0])
        self.features[3].set_mask(masks[1], masks2[1])
        self.features[6].set_mask(masks[2], masks2[2])
        self.features[8].set_mask(masks[3], masks2[3])
        self.features[10].set_mask(masks[4], masks2[4])
        self.classifier[1].set_mask(masks[5], masks2[5])
        self.classifier[4].set_mask(masks[6], masks2[6])
        self.classifier[6].set_mask(masks[7], masks2[7])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x 
    
def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model