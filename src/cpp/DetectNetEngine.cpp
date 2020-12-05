//
// Created by hao on 2020/12/2.
//
// Some code is adapted from https://github.com/NVIDIA/retinanet-examples/blob/master/csrc/engine.cpp

#include "DetectNetEngine.h"
#include "NumCpp.hpp"

class Logger : public ILogger {
public:
    explicit Logger(bool verbose)
            : _verbose(verbose) {
    }

    void log(Severity severity, const char *msg) override {
        if (_verbose || (severity != Severity::kINFO) && (severity != Severity::kVERBOSE))
            cout << msg << endl;
    }

private:
    bool _verbose{false};
};

DetectNetEngine::DetectNetEngine(const string& modelPath, int modelWidth, int modelHeight)
{
    _modelPath = modelPath;
    _modelWidth = modelWidth;
    _modelHeight = modelHeight;
    _modelSize = Size(modelWidth, modelHeight);

    Logger logger(true);
    this->_runtime = createInferRuntime(logger);
    LoadEngine(modelPath);

    buffers = vector<void*>();
    outputBuffers = vector<void*>();
    PrepareContext();
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
    auto img = PreProcess(image);

//    _context.get
//    void* mem;
//    cudaMallocHost(&mem, )

//    cudaMemcpyAsync(buffers[inputIndex0], inputData, batch_size * INPUT_D * sizeof(float), cudaMemcpyHostToDevice, _stream);

}
