//
// Created by Hao Su on 2021/1/24.
//

#include <map>
#include <vector>
#include "Yolo5Engine.h"

using namespace cv;

Yolo5Engine::Yolo5Engine(const string &modelPath, int modelWidth, int modelHeight, float nmsThreshold)
    : TrtEngine(modelPath, modelWidth, modelHeight)
{
    this->NMS_Threshold = nmsThreshold;
}

void Yolo5Engine::PreProcess(const Mat &img) {
    // letter box resize
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (img.cols*1.0);
    float r_h = Yolo::INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    int i = 0;
    for (int row = 0; row < Yolo::INPUT_H; ++row) {
        uchar* uc_pixel = out.data + row * out.step;
        for (int col = 0; col < Yolo::INPUT_W; ++col) {
            hostBuffers[0][i] = (float)uc_pixel[2] / 255.0;
            hostBuffers[0][i + Yolo::INPUT_H * Yolo::INPUT_W] = (float)uc_pixel[1] / 255.0;
            hostBuffers[0][i + 2 * Yolo::INPUT_H * Yolo::INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

vector<Yolo::Detection>
Yolo5Engine::PostProcess(float confidenceThreshold, int originWidth, int originHeight) {
    std::vector<Yolo::Detection> res;
    nms(res, hostBuffers[1], confidenceThreshold, 0.45);
    return res;
}

vector<Yolo::Detection> Yolo5Engine::DoInfer(const Mat &image, float confidenceThreshold) {
    PreProcess(image);

    cudaMemcpyAsync(deviceBuffers[0], hostBuffers[0], buffersSizeInBytes[0], cudaMemcpyHostToDevice, _stream);
    _context->enqueue(1, reinterpret_cast<void **>(deviceBuffers.data()), _stream, nullptr);
    cudaMemcpyAsync(hostBuffers[1], deviceBuffers[1], buffersSizeInBytes[1], cudaMemcpyDeviceToHost, _stream);
    cudaStreamSynchronize(_stream);

    return PostProcess(confidenceThreshold, image.cols, image.rows);
}
