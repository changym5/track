#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"

#include "FeatureTensor.h"
#include "BYTETracker.h" //bytetrack
#include "tracker.h"//deepsort


const int nn_budget=100;
const float max_cosine_distance=0.2;

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}


void run_deepsort(cv::Mat& frame, std::vector<Detection>& results,tracker& mytracker)
{
    std::vector<Detection> objects;

    DETECTIONS detections;
    for (Detection dr : results)
    {
        //cv::putText(frame, classes[dr.classId], cv::Point(dr.box.tl().x+10, dr.box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(0, 255, 0));
        if(dr.classId == 0) //person
        {
            objects.push_back(dr);
            cv::rectangle(frame, dr.box, cv::Scalar(255, 0, 0), 2);
            get_detections(DETECTBOX(dr.box.x, dr.box.y,dr.box.width,  dr.box.height),dr.conf,  detections);
        }
    }

    std::cout<<"begin track"<<std::endl;
    if(FeatureTensor::getInstance()->getRectsFeature(frame, detections))
    {
        std::cout << "get feature succeed!"<<std::endl;
        mytracker.predict();
        mytracker.update(detections);
        std::vector<RESULT_DATA> result;
        for(Track& track : mytracker.tracks) {
            if(!track.is_confirmed() || track.time_since_update > 1) continue;
            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
        }
        for(unsigned int k = 0; k < detections.size(); k++)
        {
            DETECTBOX tmpbox = detections[k].tlwh;
            cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
            cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
            // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B

            for(unsigned int k = 0; k < result.size(); k++)
            {
                DETECTBOX tmp = result[k].second;
                cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                std::string label = cv::format("%d", result[k].first);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
            }
        }
    }
    std::cout<<"end track"<<std::endl;
}



void run_bytetrack(cv::Mat& frame, std::vector<Detection>& results,BYTETracker& tracker)
{
    std::vector<STrack> output_stracks = tracker.update(results);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
        cv::putText(frame, cv::format("%d %d %.2f", output_stracks[i].track_id, output_stracks[i].cls_id, output_stracks[i].score), cv::Point(tlwh[0], tlwh[1] - 5), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
    }
}


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("source", 'i', "Video source to be detected.", true, "drones.mp4");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string source = cmd.get<std::string>("source");
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    int fps = 10;
    BYTETracker bytetracker(fps, 30);
    tracker mytracker(max_cosine_distance, nn_budget);

    YOLODetector detector {nullptr};
    

    detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
    std::cout << "Model was initialized." << std::endl;

    cv::VideoCapture capture(source);
    // image = cv::imread(imagePath);
    std::vector<Detection> result;

    while (true)
    {
        cv::Mat image;
        if (!capture.read(image))
        {
            std::cout << "read error" << std::endl;
            break;
        }
        result = detector.detect(image, confThreshold, iouThreshold);


        // track
        run_bytetrack(image, result, bytetracker);
        // run_deepsort(image, result, mytracker);


        // utils::visualizeDetection(image, result, classNames);

        cv::imshow("result", image);
        // cv::imwrite("result.jpg", image);
        if(cv::waitKey(10) == 27) break;

    }

    return 0;
}
