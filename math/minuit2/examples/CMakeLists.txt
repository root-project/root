cmake_minimum_required(VERSION 3.1)
# This is a test of the Minuit2 CMake build system.

project(Quad1F LANGUAGES CXX)

# CMake should be able to find the Minuit2 package if you have either built it or installed it

find_package(Minuit2 CONFIG REQUIRED)

add_executable(Quad1F ../test/MnTutorial/Quad1FMain.cxx ../test/MnTutorial/Quad1F.h)

target_link_libraries(Quad1F PUBLIC Minuit2::Minuit2)
