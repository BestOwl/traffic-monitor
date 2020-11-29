import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import os
import sys
import time
import argparse
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess an image before TRT SSD inferencing."""
    img = cv2.resize(img, shape)
    img = np.asarray(img).astype(np.float32)
    img = img.transpose(2, 0, 1) / 255.0
    img = np.reshape(img,(-1,))
    print(img.shape)
    return img


class TrtTrafficCamNet(object):

    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("ssd/libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        with open(self.model, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        print(self.engine)
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size

            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def __init__(self, model, input_shape, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
        
    def detect(self, img):
        """Detect objects in the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)

        np.copyto(self.host_inputs[0], img_resized)

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)

        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]

        print(len(output))
        for i in range (0, 50):
            print(output[i])


INPUT_HW = (960, 544)
cls_dict = [
    'car',
    'bicycle',
    'person',
    'road_sign'
]

filename = "1.jpg"
result_file_name = str(filename)
img = cv2.imread(filename)
model_name ="TrafficCamNet/trafficnet_int8.engine"
traCamNet = TrtTrafficCamNet(model_name, INPUT_HW)
vis = BBoxVisualization(cls_dict)
print("start detection!")

traCamNet.detect(img)
    
print("finish!")