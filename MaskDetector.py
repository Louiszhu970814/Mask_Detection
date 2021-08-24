import os, sys
import argparse
import cv2
import time
import mxnet as mx
import numpy as np
from accuracy_evaluation import predict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import resnet18, resnext50, cnn

class MaskDetector():

    def __init__(self):
        from config_farm import configuration_10_320_20L_5scales_v2 as cfg
        self.cfg = cfg
        self.ctx = mx.cpu()
        self.symbol_file_path = './symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
        self.model_file_path = './saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'

        
    def predict_face(self):
        return predict.Predict(mxnet=mx,
                                     symbol_file_path=self.symbol_file_path,
                                     model_file_path=self.model_file_path,
                                     ctx=self.ctx,
                                     receptive_field_list=self.cfg.param_receptive_field_list,
                                     receptive_field_stride=self.cfg.param_receptive_field_stride,
                                     bbox_small_list=self.cfg.param_bbox_small_list,
                                     bbox_large_list=self.cfg.param_bbox_large_list,
                                     receptive_field_center_start=self.cfg.param_receptive_field_center_start,
                                     num_output_scales=self.cfg.param_num_output_scales)

    def face_imgs_and_location(self, frame, bboxes, ratio, original_size):
        ratio_w, ratio_h = ratio
        w, h = original_size
        bboxes = torch.tensor(bboxes)
        bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3] = bboxes[:,0]*ratio_w, bboxes[:,1]*ratio_h, bboxes[:,2]*ratio_w, bboxes[:,3]*ratio_h
        face_imgs = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] for bbox in bboxes]
        return face_imgs, bboxes


    def transform_imgs(self, face_imgs, bboxes, img_size=224):
        test_tfm = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        pic = torch.randn(1,3,img_size,img_size)
        for i in range(len(face_imgs)):
            # Convert pics into PIL formate
            Pil = face_imgs[i][...,::-1]
            Pil = Image.fromarray(Pil)
            # add one dimension to fake the batch dimension
            Pil = test_tfm(Pil).unsqueeze(0)
            pic = torch.cat((pic,Pil), dim=0)
        face_imgs = pic[1:]
        return face_imgs

    def classifier(self, model, face_imgs, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        name = []
        with torch.no_grad():
            logits = model(face_imgs.to(device))
                  
        result = logits.argmax(dim=-1).cpu().numpy().tolist()
        results = {0:'with_mask', 1:'incorrect_mask', 2:'no_mask'}
        name = [results[x] for x in result]
        return name

    def display_frame(self, frame, bboxes, name):
        for index, bbox in enumerate(bboxes):
            left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name[index], (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
        return frame




def main(opt):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Your device is ', device)
    classifier = opt.classifier
    model_img_size = opt.img_size
    model = torch.load(classifier).to(device)
    mask_detector = MaskDetector()
    face_predictor = mask_detector.predict_face()   
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
        if not ret:
            cv2.waitKey(3000)
            break

        # Use small_frame to fasten locating faces
        small_frame = cv2.resize(frame, (160, 120))
        ratio = w/160, h/120
        original_size = w, h
        bboxes, infer_time = face_predictor.predict(small_frame, resize_scale=1, score_threshold=0.6, top_k=10000, \
                                                        NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])
        if len(bboxes)==0:
            cv2.namedWindow('live', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('live', 640, 480)
            cv2.imshow('live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        face_imgs, bboxes = mask_detector.face_imgs_and_location(frame, bboxes, ratio, original_size)
        face_imgs = mask_detector.transform_imgs(face_imgs, bboxes,img_size=model_img_size)
        name = mask_detector.classifier(model, face_imgs, device=device)
        frame = mask_detector.display_frame(frame, bboxes, name)
        cv2.namedWindow('live', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('live', 640, 480)
        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    cap.release()
    cv2.destroyAllWindows()



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224, help='Your model input image size')
    parser.add_argument('--classifier', type=str, default='./classifier_model/resnet18_small.pt', help='classifier model path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


