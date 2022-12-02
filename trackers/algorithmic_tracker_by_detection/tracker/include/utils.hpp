#include <vector>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>

namespace utils {

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

std::vector<Detection> ReadDetections(std::string fileName)
{
  std::ifstream ifs; // input file stream
  std::string str;
  ifs.open(
    fileName, 
    std::ios::in
  ); // input file stream

  std::vector<Detection> detections = std::vector<Detection>();
  
  if(ifs)
  {
    while ( !ifs.eof() )
    {
      std::getline (ifs, str);

      if (str.empty()) {
        break;
      }
      
      Detection detection = ParseDetectionFromString(str);
      detections.insert(detections.end(), detection);
    }

    ifs.close();
  }

  return detections;
}

}