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
#include <nvdsinfer_context.h>

using namespace std;
using namespace cv;
using namespace nvinfer1;

class DetectNetEngine {
public:
    DetectNetEngine(const string& modelPath, int modelWidth, int modelHeight);
    ~DetectNetEngine();
    Mat PreProcess(const Mat& img);
    void PostProcess();
    void DoInfer(const Mat& image, double confidenceThreshold);

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    NvDsInferContextHandle _inferContext = nullptr;

    string _modelPath;
    int _modelWidth;
    int _modelHeight;
    cv::Size _modelSize;

    vector<void*> buffers;
    vector<void*> outputBuffers;

    void LoadEngine(const string& path);
    void PrepareContext();
};


#endif //TLT_TRAFFICDETECT_DETECTNETENGINE_H
