# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/textinput package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS Getline.h)

target_sources(Core
PRIVATE
  src/Getline_color.cxx
  src/Getline.cxx
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    inc/Getline.h
    src/Getline_color.h
)

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    src
)

target_link_libraries(Core PRIVATE textinput)

ROOT_INSTALL_HEADERS()
