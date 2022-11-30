#include <vector>
#include <iostream>
#include <string>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>  // Video write

#include "utils.hpp"

namespace tracker {

struct Track {
  Track(int startFrame) : startFrame(startFrame) {
    newInsertion = true;
    endFrame = -1;
    gap = 0;
    centerPositions = std::vector<std::tuple<double, double>>();
  }
  int startFrame;
  int endFrame;
  int gap;
  bool newInsertion;
  std::vector<std::tuple<double, double>> centerPositions;
};

bool compareTracks(Track t1, Track t2)
{
  return (t1.startFrame < t2.startFrame);
}

std::vector<std::tuple<double, double>> GetInterpolatedPositions(
  std::tuple<double, double> oldPoint,
  std::tuple<double, double> newPoint,
  int gapValue) {
    auto result = std::vector<std::tuple<double, double>>();

    double x0 = std::get<0>(oldPoint);
    double y0 = std::get<1>(oldPoint);
    double x1 = std::get<0>(newPoint);
    double y1 = std::get<1>(newPoint);

    for (int i = 1; i < gapValue + 1; i++) {
      double x = x0 + (x1 - x0) * i / (gapValue + 1);
      double y = y0 + (y1 - y0) * i / (gapValue + 1);
      result.push_back(std::make_tuple(x, y));
    }

    return result;
}

struct TrackerParameters {
  int searchRadius; // in pixels
  int minAge; // in frames
  int maxTrackGap; // in frames
  double minImageSpaceDistance; // in euclidean distance of pixel coordinates
};

class Tracker {
public:
  TrackerParameters parameters;
  Tracker(TrackerParameters parameters) {
    this->parameters = parameters;
  }

  std::vector<Track> CreateTrackingLines(std::vector<utils::Detection> detections) {
    auto tracks = std::vector<Track>();
    int previousFrame = 0;
    this->numDetections = detections.back().frameNumber; // TODO: Innacurate

    for (auto& detection : detections) {
      // std::cout << "New detection starting" << std::endl;
      bool detectionWasInserted = false;
      auto currentFrame = detection.frameNumber;
      auto distanceToLastFrame = currentFrame - previousFrame;

      if (distanceToLastFrame > 0) {
        // std::cout << "Current frame: " << currentFrame << std::endl;
        for (auto& track : tracks) {
          if (!track.newInsertion) {
            track.gap += distanceToLastFrame;
            if (track.gap > parameters.maxTrackGap) {
              track.endFrame = currentFrame;
            }
          }

          track.newInsertion = false;
        }
      }

      std::tuple<double, double> detectionCenterPosition = std::make_tuple(
          detection.x0 + (detection.x1 - detection.x0) / 2.,
          detection.y0 + (detection.y1 - detection.y0) / 2.
      );

      for (auto& track : tracks) {
        if (track.gap <= parameters.maxTrackGap
          && !(track.newInsertion)
          && utils::GetEuclidDistance(
            track.centerPositions.back(), 
            detectionCenterPosition) <= parameters.searchRadius
          && track.endFrame == -1) {
            if (track.gap > 0) {
              auto interpolatedPositions = GetInterpolatedPositions(
                track.centerPositions.back(), 
                detectionCenterPosition, 
                track.gap
              );
              track.centerPositions.insert(
                track.centerPositions.end(), 
                interpolatedPositions.begin(), 
                interpolatedPositions.end()
              );
            }

            track.centerPositions.push_back(detectionCenterPosition);
            track.newInsertion = true;
            detectionWasInserted = true;
            track.gap = 0;
          }
      }

      if (!detectionWasInserted) {
        auto newTrack = Track(currentFrame);
        newTrack.centerPositions.push_back(detectionCenterPosition);
        tracks.push_back(newTrack);
        detectionWasInserted = true;
      }

      previousFrame = currentFrame;
    }

    std::cout << "Number of tracks before deletion: " << tracks.size() << std::endl;

    for (std::vector<Track>::iterator it = tracks.begin(); it!=tracks.end();) {
      // Set unfinished tracks' last frame to the last detection frame
      if (it->endFrame == -1) {
        it->endFrame = this->numDetections;
      }

      // Remove tracks not long enough (geometrically and temporally)
      if(it->centerPositions.size() < parameters.minAge
        || utils::GetEuclidDistance(it->centerPositions.front(), it->centerPositions.back())
        < parameters.minImageSpaceDistance) {
        it = tracks.erase(it);
      } else {
        ++it;
      }
    }

    std::cout << "Number of tracks after deletion: " << tracks.size() << std::endl;
    return tracks;
  }

  void VisualizeTracks(
    std::vector<Track> tracks,
    std::string inputImageDirectoryPath,
    std::string outputDirectoryPath,
    std::string outputVideoFileName) {
    
    // get size of image for video output config
    std::string imageFileName = inputImageDirectoryPath + "/frame000000.jpg";
    std::string image_path = cv::samples::findFile(imageFileName);
    cv::Mat src = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Size S = src.size();

    // Video start
    cv::VideoWriter outputVideo;
    auto EXT = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    outputVideo.open(outputVideoFileName, EXT, 60, S, true);
    if (!outputVideo.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << std::endl;
        return;
    }

    int num_digits = 6;

    std::map<int, cv::Scalar> colorMap;  // index to clor
    for (int i = 0; i < tracks.size(); i++) {
      colorMap[i] = utils::GetRandomColor();
    }
    std::map<int, std::vector<cv::KeyPoint>> pointsMap;  // index to track points

    for (int i = 0; i < this->numDetections; i++) {
      int num_zeros = num_digits - ceil(log10(i+1));
      if (num_zeros == 6) {
        num_zeros = 5; // only necessary for the first element
      }
      std::string imageFileName = inputImageDirectoryPath + "/frame" + std::string(num_zeros, '0') + std::to_string(i) + ".jpg";
      std::string image_path = cv::samples::findFile(imageFileName);
      cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

      if(img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        break;
      }
      
      int trackIndex = 0;
      for (auto& track : tracks) {
        if (track.startFrame <= i && track.endFrame >= i) {
          if (track.centerPositions.size() > i - track.startFrame) {
            auto point = std::get<0>(track.centerPositions[i - track.startFrame]);
            auto point2 = std::get<1>(track.centerPositions[i - track.startFrame]);
            pointsMap[trackIndex].push_back(cv::KeyPoint(point, point2, 10));
          }
        }
        trackIndex++;
      }

      for (int i = 0; i < tracks.size(); i++) {
        cv::drawKeypoints(img, pointsMap[i], img, colorMap[i], cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      }

      std::string outputFileName = outputDirectoryPath + "/frame" + std::string(num_zeros, '0') + std::to_string(i) + ".jpg";
      cv::imwrite(outputFileName, img);

      // Video write
      outputVideo.write(img);
    }

    // Video save
    outputVideo.release();
  }
private:
  int numDetections;
};

}