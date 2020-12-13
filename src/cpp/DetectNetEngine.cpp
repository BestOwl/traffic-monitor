//
// Created by hao on 2020/12/2.
//
// Some code is adapted from https://github.com/NVIDIA/retinanet-examples/blob/master/csrc/engine.cpp

#include "DetectNetEngine.h"

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

#define __DetectionClassNum 4

DetectNetEngine::DetectNetEngine(const string& modelPath, int modelWidth, int modelHeight, int stride, float boxNorm)
{
    _modelPath = modelPath;
    _modelWidth = modelWidth;
    _modelHeight = modelHeight;
    _modelSize = Size(modelWidth, modelHeight);
    Stride = stride;
    BoxNorm = boxNorm;

    _gridH = _modelHeight / Stride;
    _gridW = _modelWidth / Stride;
    _gridSize = _gridH * _gridW;

    Logger logger(true);
    this->_runtime = createInferRuntime(logger);
    LoadEngine(modelPath);

    deviceBuffers = vector<void*>();
    hostOutputBuffers = vector<float*>();
    buffersSize = vector<size_t>();
    buffersSizeBytes = vector<size_t>();

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

    for (auto p : hostOutputBuffers)
    {
        free(p);
    }
    hostOutputBuffers.clear();

    for (auto p : deviceBuffers)
    {
        cudaFree(p);
    }
    deviceBuffers.clear();
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
        size_t sz = maxBatchSize * dim.d[0] * dim.d[1] * dim.d[2];
        void* deviceBuf;
        cudaMalloc(&deviceBuf, sz * sizeof(float));

        deviceBuffers.push_back(deviceBuf);
        buffersSize.push_back(sz);
        buffersSizeBytes.push_back(sz * sizeof(float));

        if (_engine->bindingIsInput(i))
        {
            cout << "input[" << i << "]: " << sz << endl;
        }
        else
        {
            float* hostBuf = (float*) malloc(sz * sizeof(float));
            hostOutputBuffers.push_back(hostBuf);

            cout << "output[" << i << "]: " << sz << endl;
        }
    }
}

// Adapted from https://github.com/AlexeyAB/yolo2_light/issues/25
vector<Mat> DetectNetEngine::PreProcess(const Mat& img) {
    Mat imgResized;
    Mat imgFloat;

    resize(img, imgResized, _modelSize);
    imgResized.convertTo(imgFloat, CV_32FC3, 1.0/255, 0); //uint8 -> float, divide by 255

    Mat c;
    Mat h;
    Mat w;
    extractChannel(imgFloat, c, 0);
    extractChannel(imgFloat, h, 1);
    extractChannel(imgFloat, w, 2);

    vector<Mat> ret;
    ret.push_back(c);
    ret.push_back(h);
    ret.push_back(w);

    return ret;
}

vector<DetectedObject> DetectNetEngine::PostProcess(float* bbox, float* cov, float confidenceThreshold, int originWidth, int originHeight)
{
    std::vector<DetectedObject> objectList;

    float gridCentersX[_gridW];
    float gridCentersY[_gridH];

    for (int i = 0; i < _gridW; i++)
    {
        gridCentersX[i] = (float)(i * Stride + 0.5) / (float) BoxNorm;
    }
    for (int i = 0; i < _gridH; i++)
    {
        gridCentersY[i] = (float)(i * Stride + 0.5) / (float) BoxNorm;
    }

    for (int c = 0; c < __DetectionClassNum; c++)
    {
        float *outputX1 = bbox + (c * 4 * _gridSize);
        float *outputY1 = outputX1 + _gridSize;
        float *outputX2 = outputY1 + _gridSize;
        float *outputY2 = outputX2 + _gridSize;

        for (int h = 0; h < _gridH; h++)
        {
            for (int w = 0; w < _gridW; w++)
            {
                int i = w + h * _gridW;
                if (cov[c * _gridSize + i] >= confidenceThreshold)
                {

                    DetectedObject object;
                    object.classId = c;
                    object.confidence = cov[c * _gridSize + i];

                    float rectX1f, rectY1f, rectX2f, rectY2f;

                    rectX1f = (outputX1[w + h * _gridW] - gridCentersX[w]) * -BoxNorm;
                    rectY1f = (outputY1[w + h * _gridW] - gridCentersY[h]) * -BoxNorm;
                    rectX2f = (outputX2[w + h * _gridW] + gridCentersX[w]) * BoxNorm;
                    rectY2f = (outputY2[w + h * _gridW] + gridCentersY[h]) * BoxNorm;

                    cout << rectX1f << ", " << rectY1f << ";  " << rectX2f << ", " << rectY2f << endl;

                    // rescale to the origin image coordinates
                    float x_scale = (float) originWidth / _modelWidth;
                    float y_scale = (float) originHeight / _modelHeight;
                    BBoxCoordinate b = { };
                    b.xMin = (int) rectX1f * x_scale;
                    b.yMin = (int) rectY1f * y_scale;
                    b.xMax = (int) rectX2f * x_scale;
                    b.yMax = (int) rectY2f * y_scale;
                    object.bbox = b;

                    objectList.push_back(object);
                }
            }
        }
    }

    return objectList;
}

vector<DetectedObject> DetectNetEngine::DoInfer(const Mat& image, float confidenceThreshold) {
    auto img = PreProcess(image);

    size_t per = buffersSizeBytes[0] / 3;

    cudaMemcpyAsync(deviceBuffers[0], img[0].data, per, cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(deviceBuffers[0] + per, img[1].data, per, cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(deviceBuffers[0] + per * 2, img[2].data, per, cudaMemcpyHostToDevice, _stream);

    _context->enqueue(1, deviceBuffers.data(), _stream, nullptr);

    cudaMemcpyAsync(hostOutputBuffers[0], deviceBuffers[1], buffersSizeBytes[1], cudaMemcpyDeviceToHost, _stream);
    cudaMemcpyAsync(hostOutputBuffers[1], deviceBuffers[2], buffersSizeBytes[2], cudaMemcpyDeviceToHost, _stream);
    cudaStreamSynchronize(_stream);

    img.clear();
    return PostProcess(hostOutputBuffers[0], hostOutputBuffers[1], confidenceThreshold, image.cols, image.rows);
}
