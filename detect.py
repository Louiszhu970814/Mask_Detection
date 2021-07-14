"""LFFD Demo."""
import os, sys
import argparse
import cv2
import time
import mxnet as mx
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

sys.path.append("..")
from accuracy_evaluation import predict
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         #input size:[batch_size, 3, 224, 224]
#         #CNN dimension 1+[(N-K+2*Pad)/stride]    N:Input dimension K: kernal dimension
#         #Maxpool2D(2,2) equals to K=2, stride=2  
#         self.cnn_layers = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1),      #[32,128,128]
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),          #[32,128,128]  

#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),          #[64,64,64]

#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(4, 4, 0),          #[128,8,8]

#             nn.AdaptiveAvgPool2d((1, 1)),
      
#         )
        
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256,3)

#         )

#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.flatten(1)
#         x = self.fc_layers(x)
#         return x

def parse_args():
    parser = argparse.ArgumentParser(description='LFFD Demo.')
    parser.add_argument('--version', type=str, default='v2',
                        help='The version of pretrained model, now support "v1" and "v2".')
    parser.add_argument('--mode', type=str, default='live',
                        help='The format of input data, now support "image" of jpg and "video" of mp4.')
    parser.add_argument('--use-gpu', type=bool, default=False,
                        help='Default is cpu.')
    parser.add_argument('--data', type=str, default='./data',
                        help='The path of input and output file.')
    parser.add_argument('--model', type=str, default='./classifier_model/resnet18.pt',
                        help='The path of classifier model')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = f"cuda:{args.device}"

    model = torch.load(args.model).to(device)
    # context list
    if args.use_gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    if args.version == 'v1':
        from config_farm import configuration_10_320_20L_5scales_v1 as cfg

        symbol_file_path = './symbol_farm/symbol_10_560_25L_8scales_v1_deploy.json'
        model_file_path = './saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params'
    elif args.version == 'v2':
        from config_farm import configuration_10_320_20L_5scales_v2 as cfg

        symbol_file_path = './symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
        model_file_path = './saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
    else:
        raise TypeError('Unsupported LFFD Version.')

    face_predictor = predict.Predict(mxnet=mx,
                                     symbol_file_path=symbol_file_path,
                                     model_file_path=model_file_path,
                                     ctx=ctx,
                                     receptive_field_list=cfg.param_receptive_field_list,
                                     receptive_field_stride=cfg.param_receptive_field_stride,
                                     bbox_small_list=cfg.param_bbox_small_list,
                                     bbox_large_list=cfg.param_bbox_large_list,
                                     receptive_field_center_start=cfg.param_receptive_field_center_start,
                                     num_output_scales=cfg.param_num_output_scales)

    if args.mode == 'image':
        # data_folder = args.data
        # file_name_list = [file_name for file_name in os.listdir(data_folder) \
        #                   if file_name.lower().endswith('jpg')]

        # for file_name in file_name_list:
        #     im = cv2.imread(os.path.join(data_folder, file_name))

        #     bboxes, infer_time = face_predictor.predict(im, resize_scale=1, score_threshold=0.6, top_k=10000, \
        #                                                 NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

        #     for bbox in bboxes:
        #         cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        #     # if max(im.shape[:2]) > 1600:
        #     #     scale = 1600/max(im.shape[:2])
        #     #     im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        #     cv2.imshow('im', im)
        #     cv2.waitKey(5000)
        #     cv2.imwrite(os.path.join(data_folder, file_name.replace('.jpg', '_result.png')), im)
        print("Still developing, not done yet")
    elif args.mode == 'video':
        # win_name = 'LFFD DEMO'
        # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # data_folder = args.data
        # file_name_list = [file_name for file_name in os.listdir(data_folder) \
        #                   if file_name.lower().endswith('mp4')]
        # for file_name in file_name_list:
        #     out_file = os.path.join(data_folder, file_name.replace('.mp4', '_v2_gpu_result.avi'))
        #     cap = cv2.VideoCapture(os.path.join(data_folder, file_name))
        #     vid_writer = cv2.VideoWriter(out_file, \
        #                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, \
        #                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        #                                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        #     while cv2.waitKey(1) < 0:
        #         ret, frame = cap.read()
        #         if ret:
        #             h, w, c = frame.shape

        #         if not ret:
        #             print("Done processing of %s" % file_name)
        #             print("Output file is stored as %s" % out_file)
        #             cv2.waitKey(3000)
        #             break

        #         tic = time.time()
        #         bboxes, infer_time = face_predictor.predict(frame, resize_scale=1, score_threshold=0.6, top_k=10000, \
        #                                                     NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])
        #         toc = time.time()
        #         detect_time = (toc - tic) * 1000

        #         face_num = 0
        #         for bbox in bboxes:
        #             face_num += 1
        #             cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        #         computing_platform = 'Computing platform: NVIDIA GPU FP32'
        #         cv2.putText(frame, computing_platform, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #         input_resolution = 'Network input resolution: %sx%s' % (w, h)
        #         cv2.putText(frame, input_resolution, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #         infer_time_info = 'Inference time: %.2f ms' % (infer_time)
        #         cv2.putText(frame, infer_time_info, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #         infer_speed = 'Inference speed: %.2f FPS' % (1000 / infer_time)
        #         cv2.putText(frame, infer_speed, (5, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #         face_num_info = 'Face num: %d' % (face_num)
        #         cv2.putText(frame, face_num_info, (5, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        #         vid_writer.write(frame.astype(np.uint8))
        #         # cv2.imshow(win_name, frame)

        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

        #     cap.release()
        #     cv2.destroyAllWindows()
        print("Still developing, not done yet")
    elif args.mode == 'live':
       
        data_folder = args.data
        
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            if ret:
                h, w, c = frame.shape
            if not ret:
                cv2.waitKey(3000)
                break

            tic = time.time()           
            bboxes, infer_time = face_predictor.predict(small_frame, resize_scale=1, score_threshold=0.6, top_k=10000, \
                                                        NMS_threshold=0.4, NMS_flag=True, skip_scale_branch_list=[])

            if len(bboxes)==0:
                cv2.namedWindow('live', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('live', 640, 480)
                cv2.imshow('live', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            toc = time.time()
            detect_time = (toc - tic) * 1000
            face_num = 0
            face_locate = []
            imgs = [frame[int(bbox[1]-25)*4:int(bbox[3]+10)*4, int(bbox[0]-15)*4:int(bbox[2]+15)*4] for bbox in bboxes]
            img_size = 224
            pic = torch.randn(1,3,img_size,img_size)
            
            for i in range(len(imgs)):
                imgs[i] = Image.fromarray(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
                imgs[i] = test_tfm(imgs[i]).unsqueeze(0)
                pic = torch.cat((pic,imgs[i]), dim=0)
 
            imgs = pic[1:]
            with torch.no_grad():
                logits = model(imgs.to(device))
            
            
            result = logits.argmax(dim=-1).cpu().numpy().tolist()
            results = {0:'with_mask', 1:'incorrect_mask', 2:'no_mask'}
            name = [results[x] for x in result]

            for index, bbox in enumerate(bboxes):
                face_num += 1
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                left, top, right, bottom = int(bbox[0]-10)*4, int(bbox[1]-20)*4, int(bbox[2]+10)*4, int(bbox[3]+5)*4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # cv2.rectangle(frame, (left, bottom + 10), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name[index], (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
            

            cv2.namedWindow('live', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('live', 640, 480)
            cv2.imshow('live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print('detectin time is %f ms' % (detect_time))
        cap.release()
        cv2.destroyAllWindows()
    else:
        raise TypeError('Unsupported File Format.')


if __name__ == '__main__':
    main()
