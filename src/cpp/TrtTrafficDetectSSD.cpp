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
#include <fstream>

#include "TrtEngine.h"
#include "SSDRes18Engine.h"

using namespace std;
using namespace chrono;
using namespace cv;

void PrintHelp();
int PrintBadArguments();
int DetectPicture(const string& inputPath, const string& modelPath, const string& outputPath = "");
int DetectVideo(const string& inputPath, const string& modelPath);
void DrawRect(Mat& img, DetectedObject obj);

string classes_dict[] = { "byclce", "car", "person", "road_sign"};

/**
 * Run trt inference
 *
 * arguments:
 *  0. executable
 *  1. task_flag - 0 indicates picture; 1 indicates video
 *  2. input file path
 *  3. model TRT engine file path
 *  4. [Optional] output label path
 */
int main(int argc, char* argv[])
{
    if (argc != 4 && argc != 5)
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
        string out;
        if (argc == 5)
        {
            out = argv[4];
        }
        return DetectPicture(argv[2], argv[3], out);
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

int DetectPicture(const string& inputPath, const string& modelPath, const string& outputPath)
{
    string image_path = samples::findFile(inputPath);
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    SSDRes18Engine inferer(modelPath, 1248, 384);

    auto objects = inferer.DoInfer(img, 0.3);
    ofstream outputFile;
    if (!outputPath.empty())
    {
        outputFile.open(outputPath);
    }
    for (auto obj : objects)
    {
        DrawRect(img, obj);
        outputFile << classes_dict[obj.classId] << " " << obj.confidence << " " << obj.bbox.xMin << " " << obj.bbox.yMin << " " << obj.bbox.xMax << " " << obj.bbox.yMax << endl;
    }
    outputFile.close();

    objects.clear();
    imwrite("result.jpg", img);

    return 0;
}

inline bool exists_test0 (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

int DetectVideo(const string& inputPath, const string& modelPath)
{
    if (!fileExist(inputPath))
    {
        cout << "Could not open video file: file dose not exist" << endl;
    }
    VideoCapture video(inputPath);
    double frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
    double frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter writer("result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), 30, Size(frameWidth, frameHeight));

    SSDRes18Engine inferer(modelPath, 1248, 384);

    cout << "Start detection!" << endl;
    double currentFps;
    double fps;
    auto totalStart = system_clock::now();
    auto tick = totalStart;
    Mat frame;
    while (video.read(frame))
    {
        auto objects = inferer.DoInfer(frame, 0.3);
        for (auto obj : objects)
        {
            DrawRect(frame, obj);
        }
        writer.write(frame);

        auto tock = system_clock::now();
        currentFps = 1 / ((duration<float>)(tock - tick)).count();
        if (fps == 0)
        {
            fps = currentFps;
        }
        else
        {
            fps = fps * 0.95 + currentFps * 0.05;
        }
        tick = tock;
        cout << "\r  fps: " << fps << " " << std::flush;
    }
    cout << endl;
    auto totalEnd = system_clock::now();
    cout << endl << "Finish!" << endl;
    cout << "Time elapsed: " << ((duration<float>)(totalEnd - totalStart)).count() << " seconds";
    video.release();

    return 0;
}

void DrawRect(Mat& img, DetectedObject obj)
{
    Point topLeft(obj.bbox.xMin, obj.bbox.yMin);
    Point bottomRight(obj.bbox.xMax, obj.bbox.yMax);
    Scalar color;
    switch (obj.classId) {
        case 0:
            color = Scalar(5, 250, 90);
            break;
        case 1:
            color = Scalar(5, 250, 235);
            break;
        case 2:
            color = Scalar(250, 250, 5);
            break;
        case 3:
            color = Scalar(140, 5, 250);
            break;
    }
    rectangle(img, topLeft, bottomRight, color);
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