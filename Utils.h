#pragma once

#include "Calibrator.h"

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <iostream>

bool replace(std::string& str, const std::string& from, const std::string& to);

namespace charuco_calibration
{
    //bool readDetectorParameters(cv::Ptr<cv::aruco::DetectorParameters> &params);
    //bool readCalibrationFlags(int& calibrationFlags);
    void SetCalibratorParams(Calibrator& calibrator);
    std::ostream& saveCameraInfo(std::ostream& output, CalibrationResult& result);
    //void setLoggerName(const std::string& loggerName);
}