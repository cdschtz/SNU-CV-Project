#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace utils {

std::tuple<double, double> TOP_LEFT_CORNER = std::make_tuple(590., 225.);
std::tuple<double, double> BOTTOM_RIGHT_CORNER = std::make_tuple(1084., 485.);

bool IsInSearchArea(std::tuple<double, double> detectionCenter) {
  double xCoord = std::get<0>(detectionCenter);
  double yCoord = std::get<1>(detectionCenter);
  if (xCoord < std::get<0>(TOP_LEFT_CORNER)
      || xCoord > std::get<0>(BOTTOM_RIGHT_CORNER)
      || yCoord < std::get<1>(TOP_LEFT_CORNER)
      || yCoord > std::get<1>(BOTTOM_RIGHT_CORNER)
  ) {
    return false;
  }
  return true;
}

cv::Scalar GetRandomColor() {
  cv::Scalar color(
    (double) std::rand() / RAND_MAX * 255,
    (double) std::rand() / RAND_MAX * 255,
    (double) std::rand() / RAND_MAX * 255
  );
  return color;
}

template <typename T> T GetEuclidDistance(std::tuple<T, T> a, std::tuple<T, T> b) {
  double tmp = pow((std::get<0>(a) - std::get<0>(b)), 2) + pow((std::get<1>(a) - std::get<1>(b)), 2);
  return sqrt(tmp);
}

struct Detection {
   unsigned int  frameNumber;
   int objectId;
   double x0;
   double y0;
   double x1;
   double y1;
   double confidenceScore;
};

Detection ParseDetectionFromString(std::string detectionString)
{
  std::stringstream ss(detectionString);
  std::string segment;
  std::vector<std::string> seglist;

  while(std::getline(ss, segment, ','))
  {
    seglist.push_back(segment);
  }

  Detection detection = Detection();
  detection.frameNumber = std::stoi(seglist[0]);
  detection.objectId = std::stoi(seglist[1]);
  detection.x0 = std::stof(seglist[2]);
  detection.y0 = std::stof(seglist[3]);
  detection.x1 = std::stof(seglist[4]);
  detection.y1 = std::stof(seglist[5]);
  detection.confidenceScore = std::stof(seglist[6]);

  return detection;
}

std::map<int, std::vector<Detection>> ReadDetections(std::string fileName)
{
  std::ifstream ifs; // input file stream
  std::string str;
  ifs.open(
    fileName, 
    std::ios::in
  ); // input file stream

  auto detections = std::map<int, std::vector<Detection>>();
  std::vector<Detection> currentFrameDetections = std::vector<Detection>();
  int previousFrame = 0;
  
  if(ifs)
  {
    while ( !ifs.eof() )
    {
      std::getline (ifs, str);

      if (str.empty()) {
        break;
      }
      
      Detection detection = ParseDetectionFromString(str);
      if ((detection.frameNumber - previousFrame) > 0) {
        detections.insert({previousFrame, currentFrameDetections});
        currentFrameDetections.clear();
        previousFrame = detection.frameNumber;
      }
      currentFrameDetections.insert(currentFrameDetections.end(), detection);
    }

    ifs.close();
  }

  return detections;
}

}