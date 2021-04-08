//
// Created by hao on 2020/12/17.
//

#ifndef TRAFFICDETECT_TRTENGINE_H
#define TRAFFICDETECT_TRTENGINE_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>

#include "../../tensorrtx/yolov5/yololayer.h"

#include "Core.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

class TrtEngine {
public:
    TrtEngine(const string &modelPath, int modelWidth, int modelHeight);
    ~TrtEngine();

    virtual vector<Yolo::Detection> DoInfer(const Mat& image, float confidenceThreshold) = 0;
    virtual void PreProcess(const Mat& img) = 0;
    virtual vector<Yolo::Detection> PostProcess(float confidenceThreshold, int originWidth, int originHeight) = 0;

    string _modelPath;
    int _modelWidth;
    int _modelHeight;
    cv::Size _modelSize;

protected:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    vector<char*> deviceBuffers;
    vector<float*> hostBuffers;
    vector<size_t> buffersSize;
    vector<size_t> buffersSizeInBytes;

    void LoadEngine(const string& path);
    void PrepareContext();
};

#endif //TRAFFICDETECT_TRTENGINE_H
