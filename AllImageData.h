#pragma once
#include <map>
#include <vector>
#include <opencv2/imgcodecs.hpp>

namespace charuco_calibration
{
    struct AllImageData
    {
        std::vector<std::vector<std::vector<cv::Point2f>>> allCorners;
        std::vector<std::vector<int>> allIds;
        std::vector<cv::Mat> allImgs;
        std::vector<std::map<int, cv::Point3f>> allObjectPoints;
        std::vector<std::map<int, cv::Point2f>> allImagePoints;
    };
}
