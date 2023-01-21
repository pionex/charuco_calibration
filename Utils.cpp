#include "Utils.h"

namespace
{
    std::string loggerName("utils");
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

namespace charuco_calibration
{
    void SetCalibratorParams(charuco_calibration::Calibrator& calibrator)
    {
        calibrator.params.squaresX = 9; //nh.param("squares_x", 6);
        calibrator.params.squaresY = 6; //nh.param("squares_y", 8);
        calibrator.params.squareLength = 0.040; // * 1.55; //nh.param("square_length", 0.021);
        calibrator.params.markerLength = 0.031; // * 1.55; //nh.param("marker_length", 0.013);
        calibrator.params.dictionaryId = 2; //nh.param("dictionary_id", 4);
        calibrator.params.aspectRatio = 1.0; //nh.param("aspect_ratio", 1.0);
        calibrator.params.performRefinement = true; //nh.param("perform_refinement", false);
        calibrator.params.drawHistoricalMarkers = true; //nh.param("draw_historical_markers", true);
        calibrator.applyParams();
    }

}