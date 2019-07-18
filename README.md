# ML-Structued_Class_Blind_Pruning

This repo implements the pruning experiments of the structured class blind pruning paper on AlexNet and ResNet-50 on ImageNet 

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders

## Training

To train a model, run `ImageNet_classblind_AlexNet.py` or `ImageNet_classblind_ResNet.py`, depending on the experiment you want to run. 

```bash
python ImageNet_classblind_AlexNet.py
```

This repo is still a work under-progress = Structuring the code!
