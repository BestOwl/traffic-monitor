//
// Created by hao on 2020/12/17.
// Author: Hao Su <microhaohao@gmail.com>

#include <NvInferPlugin.h>

#include "TrtEngine.h"

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

TrtEngine::TrtEngine(const string &modelPath, int modelWidth, int modelHeight)
{
    _modelPath = modelPath;
    _modelWidth = modelWidth;
    _modelHeight = modelHeight;
    _modelSize = Size(_modelWidth, _modelHeight);

    Logger logger(true);
    this->_runtime = createInferRuntime(logger);
    LoadEngine(modelPath);

    deviceBuffers = vector<char*>();
    hostBuffers = vector<float*>();
    buffersSize = vector<size_t>();
    buffersSizeInBytes = vector<size_t>();

    PrepareContext();
}

TrtEngine::~TrtEngine()
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

    for (auto p : hostBuffers)
    {
        free(p);
    }
    hostBuffers.clear();

    for (auto p : deviceBuffers)
    {
        cudaFree(p);
    }
    deviceBuffers.clear();
}

void TrtEngine::LoadEngine(const string &path) {
    if (!fileExist(path))
    {
        cout << "Error: Could not find engine file" << endl;
        exit(-1);
    }

    cout << "Loading engine file..." << endl;
    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg (0, std::ifstream::beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    cout << "Deserializing engine..." << endl;
    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
    cout << "Successfully loaded engine." << endl;

    delete[] buffer;
}

void TrtEngine::PrepareContext() {
    _context = _engine->createExecutionContext();
    cudaStreamCreate(&_stream);

    cout << "================================" << endl;
    cout << "         Engine Summary         " << endl;
    cout << "================================" << endl;
    int bindings = _engine->getNbBindings();
    int maxBatchSize = _engine->getMaxBatchSize();
    cout << "bindings: " << bindings << endl;
    cout << "maxBatchSize: " << maxBatchSize << endl;
    for (int i = 0; i < bindings; i++)
    {
        Dims dim = _engine->getBindingDimensions(i);
        size_t sz = maxBatchSize * dim.d[0] * dim.d[1] * dim.d[2];
        size_t szInBytes = sz * sizeof(float);
        buffersSize.push_back(sz);
        buffersSizeInBytes.push_back(szInBytes);

        char* deviceBuf;
        cudaMalloc(&deviceBuf, szInBytes);
        deviceBuffers.push_back(deviceBuf);

        float* hostBuf = (float*) malloc(szInBytes);
        hostBuffers.push_back(hostBuf);

        if (_engine->bindingIsInput(i))
        {
            cout << "input[" << i << "]: ";
        }
        else
        {
            cout << "output[" << i << "]: ";
        }
        cout << dim.d[0] << "x" << dim.d[1] << "x" << dim.d[2] << "    " << szInBytes << " bytes" << endl;
    }
    cout << "================================" << endl;
}

