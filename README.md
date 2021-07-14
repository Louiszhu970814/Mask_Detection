# Mask_Detection
## First Install Pytorch
```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Use requirements.txt to install the packages

```
pip install -r requirements.txt
```

## Run detect.py
```
python detect.py
```
## Use your own classifier

Put your ".pt" files in "./classifier_model" and type

```
python detect.py --model ./classifier_model/your_model.pt 
```
## Reference

https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices
https://pytorch.org/hub/pytorch_vision_resnet/
