#include <fstream>
#include <iostream>
#include <map>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <rtabmap/core/Cerealize.h>
#include <rtabmap/utilite/UConversion.h>

namespace rtabmap
{

    std::string unixTimeToString(double unixTime) 
    {
        // Convert Unix time to time_t by truncating to the nearest second
        time_t time = static_cast<time_t>(unixTime);

        // Convert time_t to tm struct for local time
        tm *ltm = localtime(&time);

        // Use stringstream to format the date and time
        std::stringstream ss;
        ss << std::put_time(ltm, "%Y-%m-%d %H:%M:%S");

        // If you want to include milliseconds
        int milliseconds = static_cast<int>((unixTime - static_cast<double>(time)) * 1000);
        ss << '.' << std::setfill('0') << std::setw(3) << milliseconds;

        // Return the formatted string
        return ss.str();
    }

    void cerealizeKeyPointsToSimpleFormat(const std::vector<cv::KeyPoint> &keypoints, const std::string &filepath)
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

    void cerealizeVectorToSimpleFormat(const std::vector<int> &vec, const std::string &filepath)
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

    int cerealizeLikelihood(std::string fname, const std::map<int, float> &rawLikelihood)
    {
        std::ofstream file(fname);

        // Check if file is open
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return 1;
        }

        // Iterate through the map and write each key-value pair to the file
        for (const auto& pair : rawLikelihood) {
            file << pair.first << "," << pair.second << "\n";
        }
        // Close the file
        file.close();
        return 0;
    }

    int cerealizeTransform1(std::string fname, const Transform &transform, double timestamp, int frameID)
    {

        //10,2023-11-21 13:22:14.681,0,-0,1.76905,0.0917057,1.60522,0
        float x,y,z,roll,pitch,yaw;
        transform.getTranslationAndEulerAngles(x, y, z, roll, pitch, yaw);


        std::ofstream file(fname, std::ios::app);

        // Check if file is open
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return 1;
        }

        std::string time_str = unixTimeToString(timestamp);

        file << frameID << "," << time_str << "," << roll <<  "," << pitch << "," << yaw << "," << x <<  "," << y << "," << z << "\n";

        return 0;

    }

    int cerealizeTransform(std::string fname, const Transform &transform, double timestamp, int frameID)
    {
        std::ofstream file(fname, std::ios::app);

        // Check if file is open
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return 1;
        }

        Eigen::Quaternionf q = transform.getQuaternionf();

        char str[100];
        sprintf(str, "%f %f %f %f %f %f %f %f%s\n",
							timestamp,
							transform.x(),
							transform.y(),
							transform.z(),
							q.x(),
							q.y(),
							q.z(),
							q.w(),
							(" "+uNumber2Str(frameID)).c_str());

        file << str;

        return 0;
    }

    int cerealizeLoopClosure(std::string fname, double timestamp, int frameIDFrom, int frameIDTo, LoopClosureStatus status, int accepted)
    {
        std::ofstream file(fname, std::ios::app);

        // Check if file is open
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            return 1;
        }

        char str[100];
        sprintf(str, "%d %d %f %d %d\n",
							frameIDFrom,
                            frameIDTo,
                            timestamp,
							status,
                            accepted);
        file << str;
        return 0;
    }
}