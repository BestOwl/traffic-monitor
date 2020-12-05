#include <src/cnn/nvdsinfer.h>
#include <src/cnn/nvdsinfer_dbscan.h>

// uncluster, clustered
void Cnn::assignClass(std::vector &objectList, std::vector &objectListRes,
                      int num)
{
    int objNum = objectList.size();
    if (objNum < 1)
        return;

    for (int i = 0; i < num; ++i)
    {
        objectListRes[i].classId = objectList[0].classId;
        float ax = objectListRes[i].left + objectListRes[i].width / 2;
        float ay = objectListRes[i].top + objectListRes[i].height / 2;

        float bx = objectList[0].left + objectList[0].width / 2;
        float by = objectList[0].top + objectList[0].height / 2;

        float dist = sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));

        for (int j = 1; j < objNum; ++j)
        {
            bx = objectList[j].left + objectList[j].width / 2;
            by = objectList[j].top + objectList[j].height / 2;

            float distItr = sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
            if (dist > distItr)
            {
                dist = distItr;
                objectListRes[i].classId = objectList[j].classId;
            }
        }
    }
}

void Cnn::doInference(IExecutionContext &context, float *inputData, cudaStream_t &stream)
{

    cudaMemcpyAsync(buffers[inputIndex0], inputData, batch_size * INPUT_D * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(batch_size, buffers, stream, nullptr);

    float probs[classNum * OUT_DIM_H * OUT_DIM_W];
    float boxs[classNum * 4 * OUT_DIM_H * OUT_DIM_W];

    cudaMemcpyAsync(probs, buffers[outputIndex0], classNum * OUT_DIM_H * OUT_DIM_W * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(boxs, buffers[outputIndex1], classNum * 4 * OUT_DIM_H * OUT_DIM_W * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector objectList;

    int gridW = OUT_DIM_W;
    int gridH = OUT_DIM_H;
    int gridSize = gridW * gridH;
    float gcCentersX[gridW];
    float gcCentersY[gridH];
    float bboxNormX = 35.0;
    float bboxNormY = 35.0;
    float *outputCovBuf = (float *)probs;
    float *outputBboxBuf = (float *)boxs;

    int strideX = DIVIDE_AND_ROUND_UP(INPUT_W, gridW);
    int strideY = DIVIDE_AND_ROUND_UP(INPUT_H, gridH);

    for (int i = 0; i < gridW; i++)
    {
        gcCentersX[i] = (float)(i * strideX + 0.5);
        gcCentersX[i] /= (float)bboxNormX;
    }
    for (int i = 0; i < gridH; i++)
    {
        gcCentersY[i] = (float)(i * strideY + 0.5);
        gcCentersY[i] /= (float)bboxNormY;
    }

    for (int c = 0; c < classNum; c++)
    {
        float *outputX1 = outputBboxBuf + (c * 4 * gridW * gridH);
        float *outputY1 = outputX1 + gridSize;
        float *outputX2 = outputY1 + gridSize;
        float *outputY2 = outputX2 + gridSize;

        float threshold = 0.02;

        for (int h = 0; h < gridH; h++)
        {
            for (int w = 0; w < gridW; w++)
            {
                int i = w + h * gridW;
                if (outputCovBuf[c * gridSize + i] >= threshold)
                {

                    NvDsInferObjectDetectionInfo object;
                    object.classId = c;
                    object.detectionConfidence = outputCovBuf[c * gridSize + i];

                    float rectX1f, rectY1f, rectX2f, rectY2f;

                    rectX1f = (outputX1[w + h * gridW] - gcCentersX[w]) * -bboxNormX;
                    rectY1f = (outputY1[w + h * gridW] - gcCentersY[h]) * -bboxNormY;
                    rectX2f = (outputX2[w + h * gridW] + gcCentersX[w]) * bboxNormX;
                    rectY2f = (outputY2[w + h * gridW] + gcCentersY[h]) * bboxNormY;

                    /* Clip object box co-ordinates to network resolution */
                    object.left = CLIP(rectX1f, 0, INPUT_W - 1);
                    object.top = CLIP(rectY1f, 0, INPUT_H - 1);
                    object.width = CLIP(rectX2f, 0, INPUT_W - 1) - object.left + 1;
                    object.height = CLIP(rectY2f, 0, INPUT_H - 1) - object.top + 1;

                    objectList.push_back(object);
                }
            }
        }
    }

    size_t numObjects = objectList.size();
    auto unclasteredObjectList = objectList;
    NvDsInferDBScanCluster(DBScan, &DBScanParams, &objectList[0], &numObjects);
    assignClass(unclasteredObjectList, objectList, numObjects);

    for (int i = 0; i < numObjects; ++i)
    {
        auto object = objectList[i];
        cv::Scalar color(255, 255, 0);
        if (object.classId == 0)
            color = cv::Scalar(0, 255, 0);
        else if (object.classId == 1)
            color = cv::Scalar(255, 0, 0);
        else if (object.classId == 2)
            color = cv::Scalar(0, 0, 255);

        cv::rectangle(visualize, cv::Rect(object.left, object.top, object.width, object.height), color, 2, 8, 0);
    }
}