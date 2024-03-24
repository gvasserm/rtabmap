#pragma once

#include <opencv2/opencv.hpp>

namespace rtabmap
{
    void serializeKeyPointsToSimpleFormat(const std::vector<cv::KeyPoint> &keypoints, const std::string &filepath);
    void serializeVectorToSimpleFormat(const std::vector<int> &vec, const std::string &filepath);
}