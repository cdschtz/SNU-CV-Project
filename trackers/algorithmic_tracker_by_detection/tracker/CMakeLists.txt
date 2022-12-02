cmake_minimum_required(VERSION 3.0)

project(Tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_OSX_ARCHITECTURES "arm64")

find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.11.2 REQUIRED)

configure_file(include/config.h.in ${CMAKE_BINARY_DIR}/config.h)

set( NAME_SRC
  main.cpp
)

include_directories (
  ${CMAKE_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${nlohmann_json_INCLUDE_DIRS}
  include
)

add_executable(Tracker ${NAME_SRC} ${NAME_HEADERS})
target_link_libraries(Tracker ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
