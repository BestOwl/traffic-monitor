/*
 * Created by hao on 2020/12/17.
 *
 * Author: Hao Su<microhaohao@gmail.com>
 */

#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include "TrtEngine.h"
#include "SSDRes18Engine.h"

using namespace std;
using namespace chrono;
using namespace cv;

void PrintHelp();
int PrintBadArguments();
int DetectPicture(const string& inputPath, const string& modelPath);
int DetectVideo(const string& inputPath, const string& modelPath);

/**
 * Run trt inference
 *
 * arguments:
 *  0. executable
 *  1. task_flag - 0 indicates picture; 1 indicates video
 *  2. input file path
 *  3. model TRT engine file path
 */
int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        if (argc == 2)
        {
            if (strcmp("--help", argv[1]) == 0)
            {
                PrintHelp();
                return 0;
            }
        }

        return PrintBadArguments();
    }

    if (strcmp("0", argv[1]) == 0)
    {
        return DetectPicture(argv[2], argv[3]);
    }
    else if(strcmp("1", argv[1]) == 0)
    {
        return DetectVideo(argv[2], argv[3]);
    }
    else
    {
        return PrintBadArguments();
    }
}

int DetectPicture(const string& inputPath, const string& modelPath)
{
    string image_path = samples::findFile(inputPath);
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    SSDRes18Engine inferer(modelPath, 1248, 384);

    auto start = system_clock::now();
    auto objects = inferer.DoInfer(img, 0.3);
    for (auto obj : objects)
    {
        Point topLeft(obj.bbox.xMin, obj.bbox.yMin);
        Point bottomRight(obj.bbox.xMax, obj.bbox.yMax);
        rectangle(img, topLeft, bottomRight, cv::Scalar(0, 255, 0));
    }
    auto end = system_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    cout << "time: " << double(duration.count()) * nanoseconds::period::num / nanoseconds::period::den << " (sec)" << endl;

    objects.clear();
    imwrite("result.jpg", img);

    return 0;
}

int DetectVideo(const string& inputPath, const string& modelPath)
{
    if (!std::filesystem::exists(inputPath))
    {
        cout << "Could not open video file: file dose not exist" << endl;
    }
    VideoCapture video(inputPath);
    double frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
    double frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    double fps = video.get(CAP_PROP_FPS);
    VideoWriter writer("result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(frameWidth, frameHeight));

    SSDRes18Engine inferer(modelPath, 1248, 384);

    cout << "Start detection!" << endl;
    double currentFps;
    auto totalStart = system_clock::now();
    Mat frame;
    while (video.read(frame))
    {
        auto start = system_clock::now();

        auto objects = inferer.DoInfer(frame, 0.3);
        for (auto obj : objects)
        {
            Point topLeft(obj.bbox.xMin, obj.bbox.yMin);
            Point bottomRight(obj.bbox.xMax, obj.bbox.yMax);
            rectangle(frame, topLeft, bottomRight, cv::Scalar(0, 255, 0));
        }
        writer.write(frame);

        auto end = system_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        cout << "\r fps: " << double(duration.count()) * nanoseconds::period::num / nanoseconds::period::den * 60;
    }
    auto totalEnd = system_clock::now();
    cout << endl << "Finish!" << endl;
    auto duration = duration_cast<microseconds>(totalEnd - totalStart);
    cout << "Time elapsed: " << double(duration.count()) * microseconds ::period::num / microseconds::period::den * 60 << " seconds";
    video.release();

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