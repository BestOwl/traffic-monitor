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

#include "Core.h"
#include "TrtEngine.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

class DetectNetEngine : public TrtEngine {
public:
    explicit DetectNetEngine(const string& modelPath);
    vector<DetectedObject> DoInfer(const Mat& image, float confidenceThreshold) override;

protected:
    int _gridH;
    int _gridW;
    int _gridSize;

    vector<Mat> PreProcess(const Mat &img) override;
    vector<DetectedObject> PostProcess(vector<float*> outputs, float confidenceThreshold, int originWidth, int originHeight) override;
};


#endif //TLT_TRAFFICDETECT_DETECTNETENGINE_H
