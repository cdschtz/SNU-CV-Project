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
  std::cout << "Program argument read in successfully!" << "\n";

  // Initialize tracker
  tracker::Tracker tracker = tracker::Tracker(inputPath, parameterFile);

  // Create tracks
  std::cout << "Being creating tracks..." << "\n";
  tracker.CreateTrackingLines();

  // Visualization
  std::cout << "Begin visualization:" << "\n";
  tracker.VisualizeTracks();

  // End
  std::cout << "Done." << std::endl;
}