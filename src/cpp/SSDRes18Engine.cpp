//
// Created by hao on 2020/12/17.
//

#include <NvInferPlugin.h>

#include "SSDRes18Engine.h"

SSDRes18Engine::SSDRes18Engine(const string &modelPath, int modelWidth, int modelHeight) : TrtEngine(modelPath, modelWidth, modelHeight) {

}

vector<Mat> SSDRes18Engine::PreProcess(const Mat &img) {
    Mat imgResized;
    Mat imgFloat;

    resize(img, imgResized, _modelSize);
    imgResized.convertTo(imgFloat, CV_32FC3, 1.0, 0); //uint8 -> float, divide by 255

    Mat c;
    Mat h;
    Mat w;
    extractChannel(imgFloat, c, 0);
    extractChannel(imgFloat, h, 1);
    extractChannel(imgFloat, w, 2);

    vector<Mat> ret;
    ret.push_back(c);
    ret.push_back(h);
    ret.push_back(w);

    return ret;
}

vector<DetectedObject> SSDRes18Engine::PostProcess(vector<float*> outputs, float confidenceThreshold, int originWidth, int originHeight) {
    std::vector<DetectedObject> objectList { };
    for (int i = 0; i < buffersSize[1]; i += 7)
    {
        float confidence = outputs[0][i + 2];
        if (confidence >= confidenceThreshold)
        {
            DetectedObject object = { };
            object.classId = outputs[0][i + 1];
            object.confidence = confidence;

            BBoxCoordinate b = { };
            b.xMin = (int) (outputs[0][i + 3] * (float) originWidth);
            b.yMin = (int) (outputs[0][i + 4] * (float) originHeight);
            b.xMax = (int) (outputs[0][i + 5] * (float) originWidth);
            b.yMax = (int) (outputs[0][i + 6] * (float) originHeight);
            object.bbox = b;

            objectList.push_back(object);
        }
    }

    return objectList;
}

vector<DetectedObject> SSDRes18Engine::DoInfer(const Mat& image, float confidenceThreshold) {
    auto img = PreProcess(image);

    size_t per = buffersSizeBytes[0] / 3;

    cudaMemcpyAsync(deviceBuffers[0], img[0].data, per, cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(deviceBuffers[0] + per, img[1].data, per, cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(deviceBuffers[0] + per * 2, img[2].data, per, cudaMemcpyHostToDevice, _stream);

    _context->enqueue(1, reinterpret_cast<void **>(deviceBuffers.data()), _stream, nullptr);

    cudaMemcpyAsync(hostOutputBuffers[0], deviceBuffers[1], buffersSizeBytes[1], cudaMemcpyDeviceToHost, _stream);
    cudaMemcpyAsync(hostOutputBuffers[1], deviceBuffers[2], buffersSizeBytes[2], cudaMemcpyDeviceToHost, _stream);
    cudaStreamSynchronize(_stream);

    img.clear();
    return PostProcess(hostOutputBuffers, confidenceThreshold, image.cols, image.rows);
}

