#include <filesystem>
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include "Calibrator.h"
#include "CharucoStereoCalibrator.h"
#include "Utils.h"

int main(int argc, char **argv)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    if (argc != 2)
        throw std::runtime_error ("invalid args");

    auto dir = std::filesystem::path(argv[1]);

    if (!std::filesystem::is_directory(dir))
        throw std::runtime_error ("not a dir");

    for (const auto &f : std::filesystem::directory_iterator(dir))
    {
        if (!std::filesystem::is_regular_file(f) || f.path().string().find(".png") == std::string::npos)
            continue;
        if (f.path().string().find("INTENSITY") != std::string::npos)
            continue;

        auto intensity_path_string = f.path().string();
        replace(intensity_path_string, "TRITON", "HELIOS");
        replace(intensity_path_string, "RGB", "INTENSITY");

        std::cout << intensity_path_string << std::endl;

        image_pairs.emplace_back(cv::imread(f.path()), cv::imread(intensity_path_string));

        if (image_pairs.back().first.empty() || image_pairs.back().second.empty())
            throw std::runtime_error(f.path().string() + " is empty");
    }


   cv::Mat intensity_image, rgb_image;

   charuco_calibration::Calibrator intensity_calibrator;
   charuco_calibration::Calibrator rgb_calibrator;
    charuco_calibration::SetCalibratorParams(intensity_calibrator);
    charuco_calibration::SetCalibratorParams(rgb_calibrator);

   for (const auto &img : image_pairs)
   {
       auto rr = rgb_calibrator.processImage(img.first);
       auto ir = intensity_calibrator.processImage(img.second);
       if (rr.isValid() && ir.isValid())
       {
           rgb_calibrator.addToCalibrationList(rr);
           intensity_calibrator.addToCalibrationList(ir);
       }
   }

   auto rresult = rgb_calibrator.performCalibration();
   auto iresult = intensity_calibrator.performCalibration();

   if (rresult.isValid)
   {
       std::cout << "RGB Camera Matrix: " << rresult.cameraMatrix << std::endl;
       std::cout << "RGB Distortion Matrix: " << rresult.distCoeffs << std::endl;
   }

   if (iresult.isValid)
   {
       std::cout << "Depth Camera Matrix: " << iresult.cameraMatrix << std::endl;
       std::cout << "Depth Distortion Matrix: " << iresult.distCoeffs << std::endl;
   }

   if (rresult.isValid && iresult.isValid)
   {
       std::cout << "Performing Stereo Calibration" << std::endl;
       auto F = charuco_calibration::CharucoStereoCalibrator::PerformStereoCalibration(rresult, iresult);
   }


    return 0;
}
