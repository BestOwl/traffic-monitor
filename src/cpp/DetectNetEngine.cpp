//
// Created by hao on 2020/12/2.
//
// Some code is adapted from https://github.com/NVIDIA/retinanet-examples/blob/master/csrc/engine.cpp

#include "DetectNetEngine.h"

#define __DetectionClassNum 4
#define __Stride 16
#define __BoxNorm 35.0
#define __ModelWidth 960
#define __ModelHeight 544

DetectNetEngine::DetectNetEngine(const string& modelPath) : TrtEngine(modelPath, __ModelWidth, __ModelHeight)
{
    _gridH = _modelHeight / __Stride;
    _gridW = _modelWidth / __Stride;
    _gridSize = _gridH * _gridW;
}

// Adapted from https://github.com/AlexeyAB/yolo2_light/issues/25
vector<Mat> DetectNetEngine::PreProcess(const Mat& img) {
    Mat imgResized;
    Mat imgFloat;

    resize(img, imgResized, _modelSize);
    imgResized.convertTo(imgFloat, CV_32FC3, 1.0/255, 0); //uint8 -> float, divide by 255

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

vector<DetectedObject> DetectNetEngine::PostProcess(vector<float*> outputs, float confidenceThreshold, int originWidth, int originHeight)
{
    std::vector<DetectedObject> objectList;

    float gridCentersX[_gridW];
    float gridCentersY[_gridH];

    for (int i = 0; i < _gridW; i++)
    {
        gridCentersX[i] = (float)(i * __Stride + 0.5) / (float) __BoxNorm;
    }
    for (int i = 0; i < _gridH; i++)
    {
        gridCentersY[i] = (float)(i * __Stride + 0.5) / (float) __BoxNorm;
    }

    for (int c = 0; c < __DetectionClassNum; c++)
    {
        float *outputX1 = outputs[0] + (c * 4 * _gridSize);
        float *outputY1 = outputX1 + _gridSize;
        float *outputX2 = outputY1 + _gridSize;
        float *outputY2 = outputX2 + _gridSize;

        for (int h = 0; h < _gridH; h++)
        {
            for (int w = 0; w < _gridW; w++)
            {
                int i = w + h * _gridW;
                if (outputs[1][c * _gridSize + i] >= confidenceThreshold)
                {

                    DetectedObject object;
                    object.classId = c;
                    object.confidence = outputs[0][c * _gridSize + i];

                    float rectX1f, rectY1f, rectX2f, rectY2f;

                    rectX1f = (outputX1[w + h * _gridW] - gridCentersX[w]) * -__BoxNorm;
                    rectY1f = (outputY1[w + h * _gridW] - gridCentersY[h]) * -__BoxNorm;
                    rectX2f = (outputX2[w + h * _gridW] + gridCentersX[w]) * __BoxNorm;
                    rectY2f = (outputY2[w + h * _gridW] + gridCentersY[h]) * __BoxNorm;

                    // rescale to the origin image coordinates
                    float x_scale = (float) originWidth / _modelWidth;
                    float y_scale = (float) originHeight / _modelHeight;
                    BBoxCoordinate b = { };
                    b.xMin = (int) (rectX1f * x_scale);
                    b.yMin = (int) (rectY1f * y_scale);
                    b.xMax = (int) (rectX2f * x_scale);
                    b.yMax = (int) (rectY2f * y_scale);
                    object.bbox = b;

                    objectList.push_back(object);
                }
            }
        }
    }

    return objectList;
}

vector<DetectedObject> DetectNetEngine::DoInfer(const Mat& image, float confidenceThreshold) {
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
