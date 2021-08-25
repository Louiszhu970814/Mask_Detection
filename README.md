# Mask_Detection
### This project is mainly about to detect if a person is wearing a face mask and weather they are wearing the mask in a correct way.
### So the work flow of 
## First Install Pytorch
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Use requirements.txt to install the packages

```
pip install -r requirements.txt
```

## Run MaskDetector.py
```
python MaskDetector.py
```
## Use your own classifier 

Put your model files in "./classifier_model" 

```
python detect.py --model ./classifier_model/your_model.pt --img-size YOUR_MODEL_INPUT_IMAGE_SIZE
```
This project will do the preporcess to the input images before feeding them to the model, which will change images from 'BGR' to 'RGB', pixel range from [0,255] to [0,1] and numpy image formate to PIL formate.

### So when you train your model, make sure the model input images are 'RGB',[0,1] and PIL formate
 


## Reference

https://github.com/YonghaoHe/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices
https://pytorch.org/hub/pytorch_vision_resnet/
