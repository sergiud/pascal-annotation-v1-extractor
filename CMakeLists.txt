cmake_minimum_required (VERSION 3.5)
project (pav1iet
  VERSION 0.1
  DESCRIPTION "PASCAL Annotation Version 1.00 Image Extraction Tool"
  LANGUAGES CXX
)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Boost 1.70 REQUIRED COMPONENTS
  chrono
  filesystem
  program_options
  thread
  NO_MODULE
)
find_package (OpenCV 4.0 REQUIRED)
find_package (TBB 2020 REQUIRED NO_MODULE)

add_executable (pav1iet
  src/adapted.hpp
  src/grammar.hpp
  src/pav1iet.cpp
)

target_link_libraries (pav1iet PRIVATE
  Boost::boost
  Boost::chrono
  Boost::filesystem
  Boost::program_options
  Boost::thread
  opencv_imgproc
  opencv_imgcodecs
  TBB::tbb
)
