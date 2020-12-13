//
// Created by hao on 2020/12/2.
//
// Some code is adapted from https://github.com/NVIDIA/retinanet-examples/blob/master/csrc/engine.cpp

#include "DetectNetEngine.h"
#include <opencv2/core.hpp>

void Log(NvDsInferContextHandle handle,
    unsigned int uniqueID, NvDsInferLogLevel logLevel, const char* logMessage,
    void* userCtx)
{
    cout << logMessage << endl;
}

DetectNetEngine::DetectNetEngine(const string& modelPath, int modelWidth, int modelHeight)
{
    _modelPath = modelPath;
    _modelWidth = modelWidth;
    _modelHeight = modelHeight;
    _modelSize = Size(modelWidth, modelHeight);

//    Logger logger(true);
//    this->_runtime = createInferRuntime(logger);
//    LoadEngine(modelPath);
//
//    buffers = vector<void*>();
//    outputBuffers = vector<void*>();
//    PrepareContext();

    NvDsInferContextInitParams param = { };
    param.maxBatchSize = 1;
    param.networkType = NvDsInferNetworkType_Detector;
    param.networkMode = NvDsInferNetworkMode_FP32;
    param.clusterMode = NVDSINFER_CLUSTER_DBSCAN;
    param.uniqueID = 1;
    param.outputBufferPoolSize = 2;
    param.networkInputFormat = NvDsInferFormat_RGB;
    param.numDetectedClasses = 4;
    param.perClassDetectionParams = new NvDsInferDetectionParams[4];
    param.

    NvDsInferDetectionParams detectionParams = { };
    detectionParams.preClusterThreshold = 0.3;
    param.perClassDetectionParams[0] = detectionParams;
    param.perClassDetectionParams[1] = detectionParams;
    param.perClassDetectionParams[2] = detectionParams;
    param.perClassDetectionParams[3] = detectionParams;

    int pathLength = modelPath.length();
    modelPath.copy(param.modelEngineFilePath, pathLength, 0);
    param.modelEngineFilePath[pathLength] = '\0';

    NvDsInferDimsCHW inferDims = { };
    inferDims.c = 3;
    inferDims.h = 544;
    inferDims.w = 960;
    param.inferInputDims = inferDims;

    NvDsInferStatus status = NvDsInferContext_Create(&_inferContext, &param, nullptr, Log);
    assert(status == NVDSINFER_SUCCESS);
}

DetectNetEngine::~DetectNetEngine()
{
    if (_stream)
    {
        cudaStreamDestroy(_stream);
    }
    if (_context)
    {
        _context->destroy();
    }
    if (_engine)
    {
        _engine->destroy();
    }
    if (_runtime)
    {
        _runtime->destroy();
    }
    if (_inferContext)
    {
        _inferContext->destroy();
    }
}

void DetectNetEngine::LoadEngine(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg (0, std::ifstream::beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

// Adapted from https://github.com/AlexeyAB/yolo2_light/issues/25
Mat DetectNetEngine::PreProcess(const Mat& img) {
    Mat imgProcessed;
    Mat mfloat;
    Mat gpu_dst;

    resize(img, imgProcessed, _modelSize);
    cvtColor(imgProcessed, imgProcessed, COLOR_BGR2RGB);

    imgProcessed.convertTo(imgProcessed, CV_32FC3, 1.0/255, 0); //uint8 -> float, divide by 255

    size_t width = mfloat.cols * mfloat.rows;
    std::vector<Mat> input_channels {
            Mat(mfloat.rows, mfloat.cols, CV_32F, gpu_dst.ptr()[0]),
            Mat(mfloat.rows, mfloat.cols, CV_32F, gpu_dst.ptr()[width]),
            Mat(mfloat.rows, mfloat.cols, CV_32F, gpu_dst.ptr()[width * 2])
    };

    split(mfloat, input_channels); //HWC -> CHW
    return gpu_dst;
}



void DetectNetEngine::PrepareContext() {
    _context = _engine->createExecutionContext();
    _context->setOptimizationProfile(0);
    cudaStreamCreate(&_stream);

    int bindings = _engine->getNbBindings();
    int maxBatchSize = _engine->getMaxBatchSize();
    cout << "bindings: " << bindings << endl;
    cout << "maxBatchSize: " << maxBatchSize << endl;
    for (int i = 0; i < bindings; i++)
    {
        Dims dim = _engine->getBindingDimensions(i);
        size_t sz = maxBatchSize * dim.d[0] * dim.d[1] * dim.d[2] * sizeof(float);
        void* buf;
        cudaMalloc(&buf, sz);

        if (_engine->bindingIsInput(i))
        {
            buffers.push_back(buf);
            cout << "input[" << i << "]: " << sz << endl;
        }
        else
        {
            outputBuffers.push_back(buf);
            cout << "output[" << i << "]: " << sz << endl;
        }
    }
}

void DetectNetEngine::DoInfer(const Mat& image, double confidenceThreshold) {
//     auto img = PreProcess(image);

//    _context.get
//    void* mem;
//    cudaMallocHost(&mem, )

//    cudaMemcpyAsync(buffers[inputIndex0], inputData, batch_size * INPUT_D * sizeof(float), cudaMemcpyHostToDevice, _stream);

    Mat img;
    resize(image, img, _modelSize);

//    NvDsInferNetworkInfo info;
//    _inferContext->getNetworkInfo(&info);
//    _inferContext->fillLayersInfo()

    void* deviceMem;
    int size = img.total() * img.elemSize();
//    cout << size << endl;
    cudaMalloc(&deviceMem, size);
    cudaError_t result = cudaMemcpy(deviceMem, img.data, size, cudaMemcpyHostToDevice);

    NvDsInferContextBatchInput input = { };
    input.inputFormat = NvDsInferFormat_BGR;
    input.numInputFrames = 1;
    input.inputFrames = &deviceMem;
//    input.returnInputFunc

    _inferContext->queueInputBatch(input);

    NvDsInferContextBatchOutput output;
    NvDsInferStatus status = _inferContext->dequeueOutputBatch(output);

//    void* hostOutputMem = malloc(32640);
//    cudaMemcpy(hostOutputMem, output.outputDeviceBuffers[0], 32640, cudaMemcpyDeviceToHost);
//    for (int i = 0; i < 32640; i++)
//    {
//        cout << ((int*) hostOutputMem)[i] << endl;
//    }

    cout << status << endl;

    NvDsInferObject* objs = output.frames->detectionOutput.objects;
    for (int i = 0; i < output.frames->detectionOutput.numObjects; i++)
    {
        cout << objs[i].label << ": (" << objs[i].left << ", " << objs[i].top << "  (" << objs[i].width << ", " << objs[i].height;
    }

    _inferContext->releaseBatchOutput(output);
}
