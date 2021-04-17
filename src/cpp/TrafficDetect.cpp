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
#include "Core.h"
#include "TrtEngine.h"
#include "Yolo5Engine.h"
#include <queue>
#include <sstream>
#include <filesystem>

#ifdef WIN32
#define _PATH_SEP "\\"
#else
#define _PATH_SEP "/"
#endif

using namespace std;
using namespace chrono;
using namespace cv;
namespace fs = std::filesystem;
using fs::path;
using fs::exists;
using fs::is_directory;
using fs::is_regular_file;

void PrintHelp();
int PrintBadArguments();
string PrepareOutputDir(const string& sourcePath, char* argv[]);
int DetectPicture(const string& inputPath, const string& outputPath = "", bool outputImage = false);
int DetectVideo(const string& inputPath, const string& modelPath);
int DetectVideo2(const string& inputPath, const string& modelPath);
int DetectDir(const string& inputPath, const string& outputPath = "");
Yolo5Engine* inferer = nullptr;

/**
 * Run inference
 *
 * arguments:
 *  0. executable
 *  1. model TRT engine file path
 *  2. task_flag - 0 indicates picture; 1 indicates video
 *  3. input file path
 *  4. [Optional] output label path
 */
int main(int argc, char* argv[])
{
    if (argc < 4)
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
    Yolo5Engine engine(argv[1], Yolo::INPUT_W, Yolo::INPUT_H);
    inferer = &engine;
    cout << inferer->_modelPath << endl;

    if (strcmp("0", argv[2]) == 0) // picture mode
    {
        return DetectPicture(argv[3], PrepareOutputDir(argc, argv));
    }
    else if(strcmp("1", argv[2]) == 0)
    {
        return DetectVideo2(argv[3], argv[1]);
    }
    else if (strcmp("2", argv[2]) == 0) {
        return DetectDir(argv[3], PrepareOutputDir(argc, argv));
    }
    else
    {
        return PrintBadArguments();
    }
}

string PrepareOutputDir(int argc, char* argv[])
{
    if (argc == 5)
    {
        path p(argv[4]);
        if (!exists(p))
        {
            fs::create_directories(p);
        }

        return argv[4];
    }
    else
    {
        stringstream ss;
        ss << argv[3] << _PATH_SEP << "out";
        string ret = ss.str();

        path p(ret);
        if (!exists(p))
        {
            fs::create_directories(p);
        }

        return ret;
    }
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

int DetectPicture(const string& inputPath, const string& outputPath, bool outputImage)
{
    Mat img = imread(inputPath, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << inputPath << std::endl;
        return 1;
    }

    cout << inferer->_modelPath << endl;
    auto objects = inferer->DoInfer(img, 0.3);
    ofstream outputFile;
    if (!outputPath.empty())
    {
        outputFile.open(outputPath);
        for (auto obj : objects)
        {
//            outputFile << classes_dict[obj.classId] << " " << obj.confidence << " " << obj.bbox.xMin << " " << obj.bbox.yMin << " " << obj.bbox.xMax << " " << obj.bbox.yMax << endl;
            Rect rect = get_rect(img, obj.bbox);
            outputFile << classes_dict[int(obj.class_id)] << " " << obj.conf << " " << rect.x << " " << rect.y << " " << rect.x + rect.width << " " << rect.y + rect.height << endl;
        }
        outputFile.close();
    }

    if (outputImage)
    {
//        for (auto obj : objects)
//        {
//            DrawRect(img, obj);
//        }
//        imwrite("result.jpg", img);
    }

    objects.clear();
    return 0;
}


int DetectVideo(const string& inputPath, const string& modelPath)
{
    path input(inputPath);
    if (!exists(input))
    {
        cout << "Could not open video file: file dose not exist" << endl;
    }
    VideoCapture video(inputPath);
    double frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
    double frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    double fps = video.get(CAP_PROP_FPS);
    double totalFrame = video.get(CAP_PROP_FRAME_COUNT);
    VideoWriter writer("result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(frameWidth, frameHeight));

    cout << "Start detection!" << endl;
    double currentFps;
    auto totalStart = system_clock::now();
    Mat frame;
    while (video.read(frame))
    {
        //auto start = system_clock::now();
        auto objects = inferer->DoInfer(frame, 0.3);
        
        for (auto obj : objects)
        {
            rectangle(frame, get_rect(frame, obj.bbox), cv::Scalar(0, 255, 0));
        }
        //auto end = system_clock::now();
        writer.write(frame);
    }
    auto totalEnd = system_clock::now();
    cout << endl << "Finish!" << endl;
    auto duration = duration_cast<seconds>(totalEnd - totalStart);
    cout << "Time elapsed: " << duration.count() << " seconds" << endl;
    cout << "Average FPS : " << totalFrame / duration.count() << endl;
    video.release();

    return 0;
}

int DetectVideo2(const string& inputPath, const string& modelPath)
{
    if (!fileExist(inputPath))
    {
        cout << "Could not open video file: file dose not exist" << endl;
    }
    VideoCapture video(inputPath);
    double frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
    double frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    double fps = video.get(CAP_PROP_FPS);
    double totalFrame = video.get(CAP_PROP_FRAME_COUNT);

    queue<Mat> frame_queue;
    queue<Mat> output_frame_queue;

    cout << "Loading video frames to memory..." << endl;
    Mat r_frame;
    auto readStart = system_clock::now();
    while (video.read(r_frame))
    {
        frame_queue.push(r_frame);
    }
    auto readEnd = system_clock::now();
    cout << "All frames have been read!" << endl;
    cout << "Read time: " << duration_cast<seconds>(readEnd - readStart).count() << " seconds" << endl;

    cout << "Start detection!" << endl;
    auto totalStart = system_clock::now();

    if (frame_queue.empty())
    {
        return 0;
    }
    Mat frame = frame_queue.front();
    frame_queue.pop();
    inferer->PreProcess(frame);
    inferer->_context->enqueue(1, reinterpret_cast<void**>(inferer->deviceBuffers.data()), inferer->_stream, nullptr);
    while (true)
    {
        cudaStreamSynchronize(inferer->_stream);
        cudaMemcpyAsync(inferer->deviceBuffers[1], inferer->hostBuffers[1], inferer->buffersSizeInBytes[1], cudaMemcpyDeviceToHost, inferer->_stream);
        cudaMemcpyAsync(inferer->hostBuffers[0], inferer->deviceBuffers[0], inferer->buffersSizeInBytes[0], cudaMemcpyHostToDevice, inferer->_stream);
        inferer->_context->enqueue(1, reinterpret_cast<void**>(inferer->deviceBuffers.data()), inferer->_stream, nullptr);

        auto result = inferer->PostProcess(0.3f, frame.cols, frame.rows);
        for (Yolo::Detection obj : result)
        {
            rectangle(frame, get_rect(frame, obj.bbox), (0, 255, 0));
        }
        //TODO: async video write
        output_frame_queue.push(frame);

        if (frame_queue.empty())
        {
            break;
        }
        frame = frame_queue.front();
        frame_queue.pop();
        inferer->PreProcess(frame);
    }
    cudaMemcpyAsync(inferer->deviceBuffers[1], inferer->hostBuffers[1], inferer->buffersSizeInBytes[1], cudaMemcpyDeviceToHost, inferer->_stream);
    cudaStreamSynchronize(inferer->_stream);

    auto totalEnd = system_clock::now();
    cout << endl << "Finish!" << endl;
    auto duration = duration_cast<seconds>(totalEnd - totalStart);
    cout << "Time elapsed: " << duration.count() << " seconds" << endl;
    cout << "Average FPS : " << totalFrame / duration.count() << endl;
    video.release();

    cout << "Writing inference result to video files" << endl;
    auto writeStart = system_clock::now();
    VideoWriter writer("result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(frameWidth, frameHeight));
    while (!output_frame_queue.empty())
    {
        writer.write(output_frame_queue.front());
        output_frame_queue.pop();
    }
    auto writeEnd = system_clock::now();
    cout << "Write time: " << duration_cast<seconds>(writeEnd - writeStart).count() << " seconds" << endl;

    return 0;
}

int DetectDir(const string& inputPath, const string& outputPath){
    
    path dir(inputPath);
    if (!exists(dir)) {
        cout << "folder does not exist" << endl;
    }
    else if (!is_directory(dir)) {
        cout << "not a folder" << endl;
    }
    //int DetectPicture(const string & inputPath, const string & modelPath, const string & outputPath, bool outputImage)
    fs::create_directories(dir);
    for (auto& p : fs::directory_iterator(dir)) {
        if (p.is_directory())
        {
            continue;
        }
        string outLabel = outputPath + _PATH_SEP + p.path().filename().string();
        outLabel = outLabel.substr(0, outLabel.find_last_of('.')) + ".txt";
        DetectPicture(p.path().string(), outLabel, false);
    }

    return 0;
}

void PrintHelp()
{
    cout << "#zone<越战越勇> TrafficDetect 2.0" << endl;
    cout << "Usage: TrafficDetect <task_flag> <input_file> <engine_file>" << endl;
    cout << "    engine_file: path to model TensorRT engine file" << endl;
    cout << "    task_flag: 0 indicates picture, 1 indicates video" << endl;
    cout << "    input_file: path to input file" << endl;
    cout << "    output_label_path: path to output"  << endl;
    cout << endl;
}

int PrintBadArguments()
{
    cout << "Bad arguments!" << endl;
    PrintHelp();
    return 1;
}