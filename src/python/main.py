import os
import sys
import time
import argparse
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from TrtTrafficCamNet import *
from TrafficClass import get_cls_dict

def detect_video(video, trt_ssd, conf_th, vis,result_file_name):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    #print(str(frame_width)+str(frame_height))
    ##定义输入编码
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWriter = cv2.VideoWriter('result.AVI', fourcc, fps, (frame_width,frame_height))
    ##开始循环检测，并将结果写到result.mp4中
    while True:
        ret,img = video.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            videoWriter.write(img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            print("\rfps: "+str(fps),end="")
        else:
            break

def detect_one(img, trt, conf_th, vis):
    full_scrn = False
    tic = time.clock()
    ##开始检测，并将结果写到result.jpg中
    boxes, confs, clss = trt.detect(img, conf_th)

    toc = time.clock()
    curr_fps = (toc - tic)

    img = vis.draw_bboxes(img, boxes, confs, clss)
    cv2.imwrite("result.jpg", img)        
    print("time: "+str(curr_fps)+"(sec)")

def main_one():    
    filename = "../../1.jpg"
    result_file_name = str(filename)
    img = cv2.imread(filename)
    cls_dict = get_cls_dict()
    model_name ="../../TrafficCamNet/trafficnet_int8.engine"
    traCamNet = TrtTrafficCamNet(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")

    detect_one(img, traCamNet, conf_th=0.30, vis=vis)

    print("finish!")

def main_loop():   
    filename = "videoplayback.mp4"
    result_file_name = str(filename)
    video = cv2.VideoCapture(filename)
    cls_dict = get_cls_dict("ssd_resnet18_traffic".split('_')[-1])
    model_name ="ssd_resnet18_traffic"
    trt_ssd = TrtSSD(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    detect_video(video, trt_ssd, conf_th=0.3, vis=vis, result_file_name=result_file_name)
    video.release()
    cv2.destroyAllWindows()
    print("\nfinish!")

def create_detect_result():    
    #filename = "test_face.jpg"
    #result_file_name = str(filename)
    dir = "val/images"
    #img = cv2.imread(filename)
    cls_dict = get_cls_dict("ssd_mobilenet_v2_signs".split('_')[-1])
    print(cls_dict)
    model_name ="ssd_mobilenet_v2_signs"
    trt_ssd = TrtSSD(model_name, INPUT_HW)
    vis = BBoxVisualization(cls_dict)
    print("start detection!")
    detect_dir(dir, trt_ssd, conf_th=0.2, vis=vis)
    cv2.destroyAllWindows()
    print("finish!")

INPUT_HW = (960, 544)
main_one()