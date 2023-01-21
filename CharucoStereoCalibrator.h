#pragma once

#include <iostream>
#include "Calibrator.h"

namespace charuco_calibration
{
    class CharucoStereoCalibrator {
    public:

        static cv::Point3f GetObjectPoint(const cv::Ptr<cv::aruco::CharucoBoard> &board, int id)
        {
            // The getChessboardSize seems to return swapped dimension based on how the markers are actually arrange
            // There is also one less charuco corner than original width;
            auto height = board->getChessboardSize().width - 1;
            auto width = board->getChessboardSize().height;

            if (id > (height * width) - 1)
                throw std::runtime_error("Invalid marker id for defined board");

            auto column = id / height;
            auto row = id % height;

            cv::Point3f point;

            point.x = column * board->getSquareLength();
            point.y = row * board->getSquareLength();
            point.z = 0;

            return point;
        }
        static cv::Mat PerformStereoCalibration(CalibrationResult &rgb_result, CalibrationResult &depth_result)
        {
            std::vector<std::vector<cv::Point3f>> all_object_points;
            std::vector<std::vector<cv::Point2f>> all_rgb_image_points;
            std::vector<std::vector<cv::Point2f>> all_depth_image_points;

            auto num_frames = rgb_result.allImageData.allImgs.size();

            if (depth_result.allImageData.allImgs.size() != num_frames)
                throw ("RGB and Depth have different Numbers of Frames");

            for (auto i = 0; i < num_frames; i++)
            {
                auto r =  rgb_result.allImageData.allImagePoints[i];
                auto d = depth_result.allImageData.allImagePoints[i];
                auto o = rgb_result.allImageData.allObjectPoints[i];

                auto &rar = all_rgb_image_points.emplace_back();
                auto &dar = all_depth_image_points.emplace_back();
                auto &oar = all_object_points.emplace_back();

                for (auto &c : r)
                {
                    const auto &marker_id = c.first;
                    const auto &rgb_marker_corner = c.second;
                    if (d.count(marker_id) && r.count(marker_id))
                    {
                        const auto &depth_marker_corner = d.at(marker_id);

                        rar.push_back(rgb_marker_corner);
                        dar.push_back(depth_marker_corner);
                        oar.push_back(GetObjectPoint(rgb_result.board, marker_id));
                    }
                    else
                    {
                        std::cout << "Marker info doesn't exist in both depth and rgb" << std::endl;
                    }
                }
            }

            cv::Mat R, T, E, F;
            auto reprojection_error = cv::stereoCalibrate(all_object_points, all_rgb_image_points, all_depth_image_points, rgb_result.cameraMatrix, rgb_result.distCoeffs, depth_result.cameraMatrix, depth_result.distCoeffs, rgb_result.imgSize, R, T, E, F, 0, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 500, 1e-6) );

            std::cout << "Stereo Reprojection Error: " << reprojection_error << std::endl;

            std::cout << "New RGB Matrix: \n " << rgb_result.cameraMatrix << std::endl;
            std::cout << "New Depth Matrix: \n " << depth_result.cameraMatrix << std::endl;


            std::cout << "R: \n" << R << std::endl;
            std::cout << "T: \n" << T << std::endl;
            std::cout << "E: \n" << E << std::endl;
            std::cout << "F: \n" << F << std::endl;

            return F;
        }

    };
}

