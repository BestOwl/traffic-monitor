//
// Created by hao on 2020/12/17.
//

#ifndef TLT_TRAFFICDETECT_SSDRES18ENGINE_H
#define TLT_TRAFFICDETECT_SSDRES18ENGINE_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Core.h"
#include "TrtEngine.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

class SSDRes18Engine : public TrtEngine {
public:
    SSDRes18Engine(const string &modelPath, int modelWidth, int modelHeight);
    vector<DetectedObject> DoInfer(const Mat& image, float confidenceThreshold) override;

protected:
    vector<Mat> PreProcess(const Mat &img) override;
    vector<DetectedObject> PostProcess(vector<float*> outputs, float confidenceThreshold, int originWidth, int originHeight) override;
};

#endif //TLT_TRAFFICDETECT_SSDRES18ENGINE_H
