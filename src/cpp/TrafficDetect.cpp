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

using namespace std;
using namespace chrono;
using namespace cv;

void PrintHelp();
int PrintBadArguments();
int DetectPicture(const string& inputPath, const string& modelPath, const string& outputPath = "", bool outputImage = false);
int DetectVideo(const string& inputPath, const string& modelPath);
int DetectVideo2(const string& inputPath, const string& modelPath);

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

    if (strcmp("0", argv[2]) == 0) // picture mode
    {
        string out;
        if (argc == 5)
        {
            out = argv[4];
        }
        return DetectPicture(argv[3], argv[1], out);
    }
    else if(strcmp("1", argv[2]) == 0)
    {
        return DetectVideo2(argv[3], argv[1]);
    }
    else
    {
        return PrintBadArguments();
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

int DetectPicture(const string& inputPath, const string& modelPath, const string& outputPath, bool outputImage)
{
    string image_path = samples::findFile(inputPath);
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    Yolo5Engine inferer(modelPath, Yolo::INPUT_W, Yolo::INPUT_H);

    auto objects = inferer.DoInfer(img, 0.3);
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
    if (!fileExist(inputPath))
    {
        cout << "Could not open video file: file dose not exist" << endl;
    }
    VideoCapture video(inputPath);
    double frameWidth = video.get(CAP_PROP_FRAME_WIDTH);
    double frameHeight = video.get(CAP_PROP_FRAME_HEIGHT);
    double fps = video.get(CAP_PROP_FPS);
    double totalFrame = video.get(CAP_PROP_FRAME_COUNT);
    VideoWriter writer("result.mp4", VideoWriter::fourcc('M', 'P', '4', 'V'), fps, Size(frameWidth, frameHeight));

    Yolo5Engine inferer(modelPath, Yolo::INPUT_W, Yolo::INPUT_H);

    cout << "Start detection!" << endl;
    double currentFps;
    auto totalStart = system_clock::now();
    Mat frame;
    while (video.read(frame))
    {
        //auto start = system_clock::now();
        auto objects = inferer.DoInfer(frame, 0.3);
        
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

    Yolo5Engine inferer(modelPath, Yolo::INPUT_W, Yolo::INPUT_H);
    queue<Mat> frame_queue;

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
    inferer.PreProcess(frame);
    inferer._context->enqueue(1, reinterpret_cast<void**>(inferer.deviceBuffers.data()), inferer._stream, nullptr);
    while (true)
    {
        cudaStreamSynchronize(inferer._stream);
        cudaMemcpyAsync(inferer.deviceBuffers[1], inferer.hostBuffers[1], inferer.buffersSizeInBytes[1], cudaMemcpyDeviceToHost, inferer._stream);
        cudaMemcpyAsync(inferer.deviceBuffers[0], inferer.hostBuffers[0], inferer.buffersSizeInBytes[0], cudaMemcpyHostToDevice, inferer._stream);
        inferer._context->enqueue(1, reinterpret_cast<void**>(inferer.deviceBuffers.data()), inferer._stream, nullptr);

        auto result = inferer.PostProcess(0.3f, frame.cols, frame.rows);
        for (Yolo::Detection obj : result)
        {
            rectangle(frame, get_rect(frame, obj.bbox), (0, 255, 0));
        }
        //TODO: async video write

        if (frame_queue.empty())
        {
            break;
        }
        frame = frame_queue.front();
        frame_queue.pop();
        inferer.PreProcess(frame);
    }
    cudaMemcpyAsync(inferer.deviceBuffers[1], inferer.hostBuffers[1], inferer.buffersSizeInBytes[1], cudaMemcpyDeviceToHost, inferer._stream);
    cudaStreamSynchronize(inferer._stream);

    auto totalEnd = system_clock::now();
    cout << endl << "Finish!" << endl;
    auto duration = duration_cast<seconds>(totalEnd - totalStart);
    cout << "Time elapsed: " << duration.count() << " seconds" << endl;
    cout << "Average FPS : " << totalFrame / duration.count() << endl;
    video.release();

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