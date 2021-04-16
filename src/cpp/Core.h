//
// Created by hao on 2020/12/17.
//

#ifndef TRAFFICDETECT_CORE_H
#define TRAFFICDETECT_CORE_H

#include <NvInfer.h>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <iostream>
#include <sstream>
#include <filesystem>

inline std::string classes_dict[] = { "bicycle", "vehicle", "pedestrian", "road_sign"};
using namespace std;
using namespace std::filesystem;

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

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void DrawRect(cv::Mat& img, DetectedObject obj);

#endif //TRAFFICDETECT_CORE_H
