import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import TrafficObject

#加载推理引擎
class TrtTrafficCamNet(object):

    #加载通过Transfer Learning Toolkit生成的推理引擎
    def _load_engine(self):
        with open(self.model, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    #通过加载的引擎，生成可执行的上下文
    def _create_context(self):
        print(self.engine)
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            ##注意：这里的host_mem需要时用pagelocked memory，以免内存被释放
            host_mem = cuda.pagelocked_empty(shape=[size], dtype=np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    #初始化引擎
    def __init__(self, model, input_shape, output_layout=7):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
        self.output_layout = output_layout
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    #释放引擎，释放GPU显存，释放CUDA流
    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    #设置图像预处理方法以及输出处理方法
    def preprocess(self, img):
        """Preprocess an image before TRT DetectNet inferencing."""
        img = cv2.resize(img, self.input_shape)
        #cv2.imwrite('1-resize.jpg', img)

        img = np.asarray(img).astype(np.float32)
        img = img.transpose(2, 0, 1) / 255.0
        return img

    def applyBoxNorm(self, o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm
        return o1, o2, o3, o4


    # def DBScanCluster(self, boxes, confs, clss):
    #     newBoxes, newConfs, newClss = boxes, confs, clss

    #     for box, conf, clss in zip(newBoxes, newConfs, newClss):
            

    def postprocess(self, outputs, conf_th, analysis_classes, originImgShape):
        """
        Postprocesses TRT DetectNet inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        model_h = 544
        model_w = 960
        stride = 16
        self.box_norm = 35.0

        grid_h = int(model_h / stride)
        grid_w = int(model_w / stride)
        grid_size = grid_h * grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []

        self.classes = [0,1,2,3]

        for i in range(grid_h):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(grid_w):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)

        boxes, confs, clss = [], [], []
        for c in range(len(self.classes)):
            if c not in analysis_classes:
                continue

            x1_idx = (c * 4 * grid_size)
            y1_idx = x1_idx + grid_size
            x2_idx = y1_idx + grid_size
            y2_idx = x2_idx + grid_size

            bbox = outputs[0]
            for h in range(grid_h):
                for w in range(grid_w):
                    i = w + h * grid_w
                    confidence = outputs[1][c * grid_size + i]
                    if confidence >= conf_th:
                        o1 = bbox[x1_idx + w + h * grid_w]
                        o2 = bbox[y1_idx + w + h * grid_w]
                        o3 = bbox[x2_idx + w + h * grid_w]
                        o4 = bbox[y2_idx + w + h * grid_w]

                        o1, o2, o3, o4 = self.applyBoxNorm(
                            o1, o2, o3, o4, w, h)

                        # rescale to the origin image coordinates
                        x_scale = originImgShape[1] / model_w
                        y_scale = originImgShape[0] / model_h
                        xmin = int(o1 * x_scale)
                        ymin = int(o2 * y_scale)
                        xmax = int(o3 * x_scale)
                        ymax = int(o4 * y_scale)

                        boxes.append((xmin, ymin, xmax, ymax))
                        confs.append(confidence)
                        clss.append(c)
        return boxes, confs, clss
        
    #利用生成的可执行上下文执行推理
    def detect(self, img, conf_th=0.3):
        """Detect objects in the input image."""
        img_resized = self.preprocess(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())
        #将处理好的图片从CPU内存中复制到GPU显存
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
        #开始执行推理任务
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)

        #将推理结果输出从GPU显存复制到CPU内存
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        
        self.stream.synchronize()

        output = [self.host_outputs[0], self.host_outputs[1]]

        self.classes = [0,1,2,3]
        return self.postprocess(output, conf_th, self.classes, img.shape)