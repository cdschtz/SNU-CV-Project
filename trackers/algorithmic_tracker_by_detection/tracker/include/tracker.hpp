#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>  // Video write
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/types.hpp>

#include "config.h"
#include "utils.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tracker {

std::string SOURCE_DIRECTORY = SOURCE_DIR;

std::string VIDEO_OUTPUT_STEM = "video";
std::string IMAGE_OUTPUT_STEM = "images";
std::string TRACKS_OUTPUT_STEM = "tracks";

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
  Tracker(fs::path inputPath, fs::path parameterFile) {
    this->inputPath = inputPath;
    this->GetImageDimensionsAndFrameCount();
    this->dataIdentifier = inputPath.stem();
    this->ReadParameters(parameterFile);
    this->ReadDetections();
    this->SetupResultsDirectory();
  }

  void CreateTrackingLines() {
    auto tracks = std::vector<Track>();
    int previousFrame = 0;

    for (auto const& [frameNumber, detections] : this->detections) {
      auto distanceToLastFrame = frameNumber - previousFrame;

      auto currentBlockedRegions = std::vector<std::tuple<double, double>>();

      for (auto& track : tracks) {
        auto candidates = std::map<int, double>();

        for (int i=0; i < detections.size(); i++) {
          std::tuple<double, double> detectionCenterPosition = std::make_tuple(
              detections[i].x0 + (detections[i].x1 - detections[i].x0) / 2.,
              detections[i].y0 + (detections[i].y1 - detections[i].y0) / 2.
          );

          if (track.gap <= parameters.maxTrackGap
                && !(track.newInsertion)
                && utils::GetEuclidDistance<double>
                  (track.centerPositions.back(), detectionCenterPosition)
                   <= parameters.searchRadius
                && track.endFrame == -1
                && utils::IsInSearchArea(detectionCenterPosition)
                && !this->IsInBlockedRegion(currentBlockedRegions, detectionCenterPosition)) {
            auto distance = utils::GetEuclidDistance<double>(
              track.centerPositions.back(), detectionCenterPosition);
            candidates.insert({i, distance});
          }
        }

        // If no matches for track, skip
        if (candidates.empty()) {
          continue;
        }

        // Get best detection match
        auto bestCandidate = *min_element(candidates.begin(), candidates.end(),
            [](const auto& l, const auto& r) { return l.second < r.second; });
        std::tuple<double, double> detectionCenterPosition = std::make_tuple(
            detections[bestCandidate.first].x0 + (detections[bestCandidate.first].x1 - detections[bestCandidate.first].x0) / 2.,
            detections[bestCandidate.first].y0 + (detections[bestCandidate.first].y1 - detections[bestCandidate.first].y0) / 2.
        );

        // block search space around detectionCenterPosition
        currentBlockedRegions.push_back(detectionCenterPosition);

        // Interpolate positions if track is large
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
        track.gap = 0;
      }

      // update to for loop over remaining detections
      for (auto& detection : detections) {
        std::tuple<double, double> detectionCenterPosition = std::make_tuple(
            detection.x0 + (detection.x1 - detection.x0) / 2.,
            detection.y0 + (detection.y1 - detection.y0) / 2.
        );

        for (auto& blockedRegion : currentBlockedRegions) {
          if (utils::GetEuclidDistance(detectionCenterPosition, blockedRegion)) {
            // If in blocked region, we don't insert detection
            continue;
          }
        }

        auto newTrack = Track(frameNumber);
        newTrack.centerPositions.push_back(detectionCenterPosition);
        tracks.push_back(newTrack);
      }

      // Update track parameters
      for (auto& track : tracks) {
        if (!track.newInsertion && track.endFrame == -1) {
          track.gap += distanceToLastFrame;
          if (track.gap > parameters.maxTrackGap) {
            track.endFrame = frameNumber;
          }
        }
        track.newInsertion = false;
      }

      previousFrame = frameNumber;
    }

    std::cout << "Number of tracks before deletion: " << tracks.size() << std::endl;

    auto lastFrameEntry = *max_element(
      this->detections.begin(), 
      this->detections.end(),
      [](const auto& l, const auto& r) { return l.first > r.first; }
    );
    for (std::vector<Track>::iterator it = tracks.begin(); it!=tracks.end();) {
      // Set unfinished tracks' last frame to the last detection frame
      if (it->endFrame == -1) {
        it->endFrame = lastFrameEntry.first;
      }

      // Remove tracks not long enough (geometrically and temporally)
      if(it->centerPositions.size() < parameters.minAge
        || utils::GetEuclidDistance<double>(it->centerPositions.front(), it->centerPositions.back())
        < parameters.minImageSpaceDistance) {
        it = tracks.erase(it);
      } else {
        ++it;
      }
    }

    std::cout << "Number of tracks after deletion: " << tracks.size() << std::endl;
    this->tracks = tracks;
  }

  void VisualizeTracks() {
    cv::VideoWriter outputVideo;
    auto EXT = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    fs::path videoFile = this->resultsDirectory / this->dataIdentifier / VIDEO_OUTPUT_STEM / "output.avi";
    if (fs::exists(videoFile)) {
      fs::remove(videoFile);
    }

    outputVideo.open(videoFile, EXT, 60, this->imageSize, true);
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

    for (int i = 0; i < this->numFrames; i++) {
      int num_zeros = num_digits - ceil(log10(i+1));
      if (num_zeros == 6) {
        num_zeros = 5; // only necessary for the first element
      }

      fs::path imagePath = this->inputPath / IMAGE_OUTPUT_STEM;
      fs::path imageFileName = imagePath / ("frame" + std::string(num_zeros, '0') + std::to_string(i) + ".jpg");
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

      fs::path outputImagePath = this->resultsDirectory / this->dataIdentifier / IMAGE_OUTPUT_STEM;
      fs::path outputFileName = outputImagePath / ("frame" + std::string(num_zeros, '0') + std::to_string(i) + ".jpg");
      cv::imwrite(outputFileName, img);

      // Video write
      outputVideo.write(img);
    }

    // Video save
    outputVideo.release();
  }
private:
  int numFrames;

  cv::Size imageSize;

  fs::path inputPath;
  fs::path resultsDirectory;
  std::string dataIdentifier;

  std::vector<utils::Detection> detections;
  std::vector<Track> tracks;

  void SetupResultsDirectory() {
    fs::path resultsDirectory = fs::path(SOURCE_DIRECTORY) / "results";
    if (!fs::exists(resultsDirectory)) {
      fs::create_directory(resultsDirectory);
    }
    this->resultsDirectory = resultsDirectory;

    // file dependent identifier
    if (!fs::exists(resultsDirectory / dataIdentifier)) {
      fs::create_directory(resultsDirectory / dataIdentifier);
      fs::create_directory(resultsDirectory / dataIdentifier / "tracks");
      fs::create_directory(resultsDirectory / dataIdentifier / "images");
      fs::create_directory(resultsDirectory / dataIdentifier / VIDEO_OUTPUT_STEM);
    } else {
      fs::path previousResultsPath = resultsDirectory / dataIdentifier;
      fs::remove_all(previousResultsPath);
    }
  }

  void ReadParameters(std::string parameterFile) {
    // Parse tracker parameters
    std::ifstream f(parameterFile);
    json jsonParameters = json::parse(f);
    tracker::TrackerParameters parameters {
      jsonParameters["searchRadius"].get<int>(),
      jsonParameters["minAge"].get<int>(),
      jsonParameters["maxTrackGap"].get<int>(),
      jsonParameters["minImageSpaceDistance"].get<double>(),
    };

    std::cout << "Initialized tracker with parameters:" << "\n";
    std::cout << "Search Radius: " << parameters.searchRadius << "\n";
    std::cout << "Min Age: " << parameters.minAge << "\n";
    std::cout << "Max Track Gap: " << parameters.maxTrackGap << "\n";
    std::cout << "Min Image Space Distance: " << parameters.minImageSpaceDistance << "\n";

    this->parameters = parameters;
  }

  void ReadDetections() {
    fs::path detectionsFile = this->inputPath / "detections" / "detections.txt";
    this->detections = utils::ReadDetections(detectionsFile);
  }

  void GetImageDimensionsAndFrameCount() {
    // Frame Count
    fs::path imagesDirectory = this->inputPath / "images";
    int numFrames = 0;
    for (auto& p : fs::directory_iterator(imagesDirectory)) {
      numFrames++;
    }
    this->numFrames = numFrames;

    // Image Dimensions
    fs::path imagePath = this->inputPath / "images" / "frame000000.jpg";
    if (!fs::exists(imagePath)) {
      std::cout << "Image does not exist: " << imagePath << std::endl;
      return;
    }
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    this->imageSize = img.size();  // necessary for video creation
  }
};

}