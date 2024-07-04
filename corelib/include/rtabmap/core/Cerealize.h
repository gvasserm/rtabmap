#pragma once

#include <opencv2/opencv.hpp>

#include <rtabmap/core/Transform.h>

#define CEREALIZE 1

namespace rtabmap
{

    enum LoopClosureStatus {
        LOCAL,
        GLOBAL    
    };

    void cerealizeKeyPointsToSimpleFormat(const std::vector<cv::KeyPoint> &keypoints, const std::string &filepath);
    void cerealizeVectorToSimpleFormat(const std::vector<int> &vec, const std::string &filepath);
    int cerealizeLikelihood(std::string fname, const std::map<int, float> &rawLikelihood);
    int cerealizeTransform(std::string fname, const Transform &transform, double timestamp, int frameID);
    int cerealizeLoopClosure(std::string fname, double timestamp, int frameIDFrom, int frameIDTo, LoopClosureStatus status, int accepted);

}