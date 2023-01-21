#include "Calibrator.h"

#include <iostream>
#include <chrono>
#include <opencv2/imgcodecs.hpp>

namespace
{
    /* Starting time point (for logging purposes) */
    std::chrono::system_clock::time_point calibStart;
}

namespace charuco_calibration
{



    void
    FIX_26082015_getBoardObjectAndImagePoints(const cv::Ptr<cv::aruco::Board> board,
                                              std::vector<cv::Point2f> _corners,
                                              std::vector<int> _ids,
                                              std::map<int, std::vector<cv::Point3f>> &objPoints,
                                              std::map<int, std::vector<cv::Point2f>> &imgPoints)
    {
        CV_Assert(board->ids.size() == board->objPoints.size());
        CV_Assert(_ids.size() == _corners.size());
        // look for detected markers that belong to the board and get their information
        for(const auto &cid : _ids)
        {
            for(const auto &bid : board->ids)
            {
                if(cid == bid)
                {
                    for(auto p = 0; p < 4; ++p)
                    {
                        objPoints[cid].push_back(board->objPoints[bid][p]);
                        imgPoints[cid].push_back(_corners[p]);
                    }
                }
            }
        }
    }

    void Calibrator::consoleLogFunction(LogLevel logLevel, const std::string& message)
    {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now - calibStart).count();
        std::string severity;
        switch(logLevel)
        {
            case DEBUG:
                severity = "[DEBUG]";
                break;
            case INFO:
                severity = "[INFO]";
                break;
            case WARN:
                severity = "[WARN]";
                break;
            case ERROR:
                severity = "[ERROR]";
                break;
            case FATAL:
                severity = "[FATAL]";
                break;
            default:
                severity = "[UNKNOWN!]";
        }

        if (logLevel < LogLevel::WARN)
        {
            std::cout << timestamp << severity << message << std::endl;
        }
        else
        {
            std::cerr << timestamp << severity << message << std::endl;
        }
    }

    Calibrator::Calibrator() : arucoDetectorParams(cv::aruco::DetectorParameters::create()), calibLogger(Calibrator::consoleLogFunction)
    {
        // Set starting time point for logging purposes
        calibStart = std::chrono::system_clock::now();
    }

/**
 * Apply new detector parameters.
 *
 * @note This should always be called after changing the params field.
 */
    void Calibrator::applyParams()
    {
        arucoDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(params.dictionaryId));
        charucoBoard = cv::aruco::CharucoBoard::create(params.squaresX, params.squaresY, params.squareLength, params.markerLength, arucoDictionary);
    }

/**
 * Find board on the provided image.
 *
 * @param image Source cv::Mat image.
 *
 * @return CalibratorDetectionResult struct with detected markers and source image.
 * @note Return value should be passed to drawDetectionResult and addToCalibrationList methods.
 */
    CalibratorDetectionResult Calibrator::processImage(const cv::Mat& image)
    {
        cv::Mat imageCopy;
        image.copyTo(imageCopy);

        static int x = 0;


        CalibratorDetectionResult result(image);
        std::vector<std::vector<cv::Point2f>> rejectedCorners;
        cv::aruco::detectMarkers(result.sourceImage,
                                 arucoDictionary, result.corners,
                                 result.ids, arucoDetectorParams,
                                 rejectedCorners);

        if (result.ids.size() > 0)
        {
            if (params.performRefinement)
            {
                cv::aruco::refineDetectedMarkers(result.sourceImage,
                                                 charucoBoard, result.corners, result.ids,
                                                 rejectedCorners);
            }

            cv::aruco::interpolateCornersCharuco(result.corners,
                                                 result.ids, result.sourceImage, charucoBoard,
                                                 result.charucoCorners, result.charucoIds);

            for (int i = 0; i < result.ids.size(); ++i)
            {
                result.imagePoints[result.charucoIds[i]] = result.charucoCorners[i];
            }

            if (!result.charucoIds.empty())
                cv::aruco::drawDetectedCornersCharuco(imageCopy, result.charucoCorners, result.charucoIds, cv::Scalar(0, 255, 0));

            cv::imwrite("/home/perception/Desktop/Charuco/out/" + std::to_string(x) + ".png", imageCopy);
            x++;

        }

        return result;
    }

/**
 * Draw detected markers on the image. Optionally show previously detected
 * markers (depending on the calibrator parameters).
 *
 * @param detectionResult Detection result, as returned by processImage().
 *
 * @return cv::Mat containing the source image with detected markers.
 */
    cv::Mat Calibrator::drawDetectionResults(const CalibratorDetectionResult& detectionResult)
    {
        cv::Mat resultImage = detectionResult.sourceImage.clone();
        if (detectionResult.isValid())
        {
            cv::aruco::drawDetectedMarkers(resultImage, detectionResult.corners);
            cv::aruco::drawDetectedCornersCharuco(resultImage,
                                                  detectionResult.charucoCorners,
                                                  detectionResult.charucoIds);
        }

        if (params.drawHistoricalMarkers)
        {
            for (const auto& frameCorner : allImageData.allCorners)
            {
                for (const auto& innerFrameCorner : frameCorner)
                {
                    for (const auto& inFrameCorner : innerFrameCorner)
                    {
                        cv::circle(resultImage, inFrameCorner, 1, cv::Scalar(255, 0, 0));
                    }
                }
            }
        }

        return resultImage;
    }

    CalibrationResult Calibrator::performCalibration()
    {
        CalibrationResult result;

        if (params.calibrationFlags & cv::CALIB_FIX_ASPECT_RATIO)
        {
            result.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
            result.cameraMatrix.at<double>(0, 0) = params.aspectRatio;
        }

        calibLogger(LogLevel::INFO, "Starting calibration with " + std::to_string(allImageData.allImgs.size()) + " images");
        if (allImageData.allImgs.size() < 1)
        {
            calibLogger(LogLevel::ERROR, "Not enough images to perform calibration");
            result.isValid = false;
            return result;
        }

        calibLogger(LogLevel::INFO, "Preparing data for ArUco calibration");
        // Assume all images have the same size
        result.imgSize = allImageData.allImgs[0].size();

        // Prepare data for ArUco calibration
        std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
        std::vector<int> allIdsConcatenated;
        std::vector<int> markerCounterPerFrame;

        markerCounterPerFrame.reserve(allImageData.allCorners.size());

        for(size_t i = 0; i < allImageData.allCorners.size(); ++i)
        {
            markerCounterPerFrame.push_back(int(allImageData.allCorners[i].size()));
            for(size_t j = 0; j < allImageData.allCorners[i].size(); ++j)
            {
                allCornersConcatenated.push_back(allImageData.allCorners[i][j]);
                allIdsConcatenated.push_back(allImageData.allIds[i][j]);
            }
        }

        calibLogger(LogLevel::INFO, "Performing ArUco calibration");
        // Calibrate camera using ArUco markers
        result.arucoReprojectionError = cv::aruco::calibrateCameraAruco(
                allCornersConcatenated, allIdsConcatenated,
                markerCounterPerFrame, charucoBoard, result.imgSize,
                result.cameraMatrix, result.distCoeffs, cv::noArray(), cv::noArray(),
                params.calibrationFlags);

        calibLogger(LogLevel::INFO, "Preparing data for ChArUco calibration");

        // Prepare data for ChArUco calibration
        int numFrames = int(allImageData.allCorners.size());
        std::vector<cv::Mat> allCharucoCorners;
        std::vector<cv::Mat> allCharucoIds;
        std::vector<cv::Mat> filteredImages;

        allCharucoCorners.reserve(numFrames);
        allCharucoIds.reserve(numFrames);


        for(size_t i = 0; i < numFrames; ++i)
        {
            calibLogger(LogLevel::INFO, "Interpolating data for image #" + std::to_string(i));
            // Interpolate corners using calculated camera parameters
            cv::Mat currentCharucoCorners, currentCharucoIds;

            cv::aruco::interpolateCornersCharuco(allImageData.allCorners[i], allImageData.allIds[i], allImageData.allImgs[i],
                                                 charucoBoard, currentCharucoCorners, currentCharucoIds,
                                                 result.cameraMatrix, result.distCoeffs);

            if (currentCharucoCorners.size().height >= 4)
            {
                allCharucoCorners.push_back(currentCharucoCorners);
                allCharucoIds.push_back(currentCharucoIds);
                filteredImages.push_back(allImageData.allImgs[i]);
            }
            else
            {
                calibLogger(LogLevel::WARN, "Rejecting image #" + std::to_string(i) + ": not enough charuco corners (found " + std::to_string(currentCharucoCorners.size().height) + ")");
            }

        }

        if (allCharucoCorners.size() < 4)
        {
            calibLogger(LogLevel::ERROR, "Could not find at least 4 ChAruco corners");
            // Not enough corners to perform calibration, bail out
            result.isValid = false;
            return result;
        }


        calibLogger(LogLevel::INFO, "Performing ChAruco calibration");
        result.reprojectionError = cv::aruco::calibrateCameraCharuco(
                allCharucoCorners, allCharucoIds, charucoBoard, result.imgSize,
                result.cameraMatrix, result.distCoeffs, result.rvecs, result.tvecs,
                params.calibrationFlags);

        calibLogger(LogLevel::INFO, "Calibration finished; reprojection error: " +
                                    std::to_string(result.reprojectionError) + ", ArUco reprojection error: " +
                                    std::to_string(result.arucoReprojectionError));

        result.isValid = true;
        result.allImageData = allImageData;
        result.board = charucoBoard;
        calibLogger(LogLevel::INFO, "All done");
        return result;
    }

}

