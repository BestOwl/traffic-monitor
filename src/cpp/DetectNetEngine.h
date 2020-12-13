//
// Created by hao on 2020/12/2.
//

#ifndef TLT_TRAFFICDETECT_DETECTNETENGINE_H
#define TLT_TRAFFICDETECT_DETECTNETENGINE_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace nvinfer1;

struct BBoxCoordinate {
    int xMin;
    int yMin;
    int xMax;
    int yMax;
};

struct DetectedObject {
    uint8_t classId;
    BBoxCoordinate bbox;
    float confidence;
};

class DetectNetEngine {
public:
    int Stride;
    float BoxNorm;

    DetectNetEngine(const string& modelPath, int modelWidth, int modelHeight, int stride = 16, float boxNorm = 35.0);
    ~DetectNetEngine();
    vector<DetectedObject> DoInfer(const Mat& image, float confidenceThreshold);

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    string _modelPath;
    int _modelWidth;
    int _modelHeight;
    cv::Size _modelSize;
    int _gridH;
    int _gridW;
    int _gridSize;

    vector<void*> deviceBuffers;
    vector<float*> hostOutputBuffers;
    vector<size_t> buffersSize;
    vector<size_t> buffersSizeBytes;

    vector<Mat> PreProcess(const Mat& img);
    vector<DetectedObject> PostProcess(float* bbox, float* cov, float confidenceThreshold, int originWidth, int originHeight);

    void LoadEngine(const string& path);
    void PrepareContext();
};


#endif //TLT_TRAFFICDETECT_DETECTNETENGINE_H
