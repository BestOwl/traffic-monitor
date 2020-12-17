//
// Created by hao on 2020/12/17.
//

#ifndef TLT_TRAFFICDETECT_CORE_H
#define TLT_TRAFFICDETECT_CORE_H

#include <NvInfer.h>
#include <string>
#include <fstream>

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

#endif //TLT_TRAFFICDETECT_CORE_H
