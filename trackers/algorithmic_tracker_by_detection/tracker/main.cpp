#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>

#include "tracker.hpp"

int main(int argc, char **argv) {
  std::cout << "\n";
  std::string inputPath = argv[1];
  std::string parameterFile = argv[2];
  std::cout << "Data file:\n" << inputPath << "\n\n";
  std::cout << "Parameter file:\n" << parameterFile << "\n\n";

  // Initialize tracker
  tracker::Tracker tracker = tracker::Tracker(inputPath, parameterFile);

  // Create tracks
  std::cout << "Begin creating tracks..." << "\n";
  tracker.CreateTrackingLines();

  // Visualization
  std::cout << "Begin visualization:" << "\n";
  tracker.VisualizeTracks(30);

  // End
  std::cout << "Done." << std::endl;
}