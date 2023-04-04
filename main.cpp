#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"

#include "FeatureTensor.h"
#include "BYTETracker.h" //bytetrack
#include "tracker.h"//deepsort

#include <librealsense2/rs.hpp>
#include <cv-helpers.hpp>

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
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
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

    // cv::VideoCapture capture(source);
    // image = cv::imread(imagePath);
    std::vector<Detection> result;

    // init camera
    std::cout << "opening camera ..." << std::endl;
    rs2::pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    std::cout << "camera is open" << std::endl;


    while (true)
    {
         // wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame,
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number)
            continue;
        last_frame_number = static_cast<int>(color_frame.get_frame_number());

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(depth_frame);


        
        result = detector.detect(color_mat, confThreshold, iouThreshold);


        // track
        run_bytetrack(color_mat, result, bytetracker);
        // run_deepsort(image, result, mytracker);


        // utils::visualizeDetection(image, result, classNames);

        cv::imshow("Tracking", color_mat);
        cv::imshow("Depth Map", depth_mat);

        // video.write(frame);
        char c = cv::waitKey(1);
        if (c == 27) // Wait for 'esc' key press to exit
        {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
