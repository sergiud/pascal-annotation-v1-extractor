cmake_minimum_required (VERSION 3.16)
project (pav1iet
  VERSION 0.1
  DESCRIPTION "PASCAL Annotation Version 1.00 Image Extraction Tool"
  LANGUAGES CXX
)

find_package (Boost 1.70 REQUIRED COMPONENTS
  program_options
  NO_MODULE
)
find_package (OpenCV 4.0 REQUIRED imgproc imgcodecs)
find_package (TBB 2021.4 REQUIRED NO_MODULE)

add_executable (pav1iet
  src/adapted.hpp
  src/grammar.hpp
  src/pav1iet.cpp
)

target_compile_features (pav1iet PRIVATE cxx_std_20)

target_link_libraries (pav1iet PRIVATE
  Boost::boost
  Boost::program_options
  opencv_imgproc
  opencv_imgcodecs
  TBB::tbb
)
