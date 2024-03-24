#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace rtabmap
{

    void serializeKeyPointsToSimpleFormat(const std::vector<cv::KeyPoint> &keypoints, const std::string &filepath)
    {
        std::ofstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
            return;
        }

        for (const auto &kp : keypoints)
        {
            file << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << "," << kp.response << "," << kp.octave << "," << kp.class_id << std::endl;
        }
    }

    void serializeVectorToSimpleFormat(const std::vector<int> &vec, const std::string &filepath)
    {
        std::ofstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
            return;
        }

        for (size_t i = 0; i < vec.size(); ++i)
        {
            file << vec[i];
            if (i < vec.size() - 1)
                file << ",";
        }
    }
}