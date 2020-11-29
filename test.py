import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda

import common

def load_engine(trt_path):
    with open(trt_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def pad2square_cv2(image):
    h,w,c = image.shape
    dim_diff = np.abs(h-w)
    pad1,pad2= dim_diff//2 ,dim_diff-dim_diff//2

    if h<=w:
        image = cv2.copyMakeBorder(image,pad1,pad2,0,0,cv2.BORDER_CONSTANT,value=0)
    else:
        image = cv2.copyMakeBorder(image,0,0,pad1,pad2,cv2.BORDER_CONSTANT,value=0)

    return image

def get_sample(img_path='./data/pics/test.jpg'):
    img = cv2.imread(img_path)
    img = pad2square_cv2(img)
    img = img/255

    img = cv2.resize(img,(960,544))
    img = img.transpose((2,0,1))
    print(img.shape)
    img = np.reshape(img,(-1,))
    
    return img


filename = "1.jpg"
result_file_name = str(filename)
img = cv2.imread(filename)
img_resized = pad2square_cv2(img)

engine = load_engine("TrafficCamNet/trafficnet_int8.engine")
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
with engine.create_execution_context() as context:
    np.copyto(inputs[0].host, img)
    res = common.do_inference(context, bindings=bindings, inputs=inputs,
                                    outputs=outputs, stream=stream)
