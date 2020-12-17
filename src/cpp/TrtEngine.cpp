//
// Created by hao on 2020/12/17.
//

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
    hostOutputBuffers = vector<float*>();
    buffersSize = vector<size_t>();
    buffersSizeBytes = vector<size_t>();

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

void TrtEngine::LoadEngine(const string &path) {
    if (!fileExist(path))
    {
        cout << "Error: Could not found engine file" << endl;
        exit(-1);
    }

    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg (0, std::ifstream::beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    initLibNvInferPlugins(getLogger(), "");
    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

void TrtEngine::PrepareContext() {
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
        char* deviceBuf;
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

