# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.1)
# This is a test of the Minuit2 CMake build system.

project(Quad1F LANGUAGES CXX)

# CMake should be able to find the Minuit2 package if you have either built it or installed it
find_package(Minuit2 CONFIG REQUIRED)

# Reusing the existing test file for simplicity
add_executable(Quad1F
    ../../test/MnTutorial/Quad1FMain.cxx
    ../../test/MnTutorial/Quad1F.h
    )

# Linking with Minuit2::Minuit2 target
target_link_libraries(Quad1F PUBLIC Minuit2::Minuit2)

# Run this executable as a test with make test
enable_testing()
add_test(NAME Quad1F COMMAND Quad1F)
