/*
 * Created by hao on 2020/12/2.
 *
 * Author: Hao Su<microhaohao@gmail.com>
 */

#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include "DetectNetEngine.h"

using namespace std;
using namespace cv;

void PrintHelp();
int PrintBadArguments();
int DetectPicture(string inputPath, string modelPath, int modelWidth, int modelHeight);
int DetectVideo(string inputPath, string modelPath, int modelWidth, int modelHeight);

/**
 * Run trt inference
 *
 * arguments:
 *  0. executable
 *  1. task_flag - 0 indicates picture; 1 indicates video
 *  2. input file path
 *  3. model TRT engine file path
 *  4. model width
 *  5. model height
 */
int main(int argc, char* argv[])
{
    if (argc != 6)
    {
        if (argc != 2 || strcmp("--help", argv[1]) == 0)
        {
            cout << "Bad arguments!" << endl;
        }
        PrintHelp();
    }

    int modelWidth = 0;
    int modelHeight = 0;
    try
    {
        modelWidth = stoi(argv[4], nullptr);
        modelHeight = stoi(argv[5], nullptr);
    }
    catch (std::invalid_argument &e) {
        return PrintBadArguments();
    }

    if (strcmp("0", argv[1]) == 0)
    {
        return DetectPicture(argv[2], argv[3], modelWidth, modelHeight);
    }
    else if(strcmp("1", argv[1]) == 0)
    {
        return DetectVideo(argv[2], argv[3], modelWidth, modelHeight);
    }
    else
    {
        return PrintBadArguments();
    }
}

int DetectPicture(string inputPath, string modelPath, int modelWidth, int modelHeight)
{
    string image_path = samples::findFile(inputPath);
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    DetectNetEngine inferer(modelPath, modelWidth, modelHeight);

    auto objects = inferer.DoInfer(img, 0.3);
    for (auto obj : objects)
    {
        Point topLeft(obj.bbox.xMin, obj.bbox.yMin);
        Point bottomRight(obj.bbox.xMax, obj.bbox.yMax);
        rectangle(img, topLeft, bottomRight, cv::Scalar(0, 255, 0));
    }

    objects.clear();
    imwrite("result.jpg", img);

    return 0;
}

int DetectVideo(string inputPath, string modelPath, int modelWidth, int modelHeight)
{
    return 0;
}

void PrintHelp()
{
    cout << "#zone<越站越勇> TrtTrafficDetect 1.0" << endl;
    cout << "Usage: TrtTrafficDetect <task_flag> <input_file> <engine_file>" << endl;
    cout << "    task_flag: 0 indicates picture, 1 indicates video" <<endl;
    cout << "    input_file: path to input file" <<endl;
    cout << "    engine_file: path to model TensorRT engine file" <<endl;
    cout << endl;
}

int PrintBadArguments()
{
    cout << "Bad arguments!" << endl;
    PrintHelp();
    return 1;
}