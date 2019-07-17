# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/qt5webdisplay package
############################################################################

if(POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()

find_package(Qt5 COMPONENTS Core WebEngine WebEngineWidgets CONFIG)

if(NOT Qt5_FOUND)
  if(fail-on-missing)
    message(FATAL_ERROR "Could NOT find Qt5")
  else()
    message(WARNING "Qt5 not found, disabling option 'qt5web'")
    set(qt5web OFF CACHE BOOL "Disabled because Qt5 not found" FORCE)
    return()
  endif()
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(SYSTEM ${Qt5WebEngine_INCLUDE_DIRS})
include_directories(SYSTEM ${Qt5WebEngineWidgets_INCLUDE_DIRS})

ROOT_OBJECT_LIBRARY(RootQt5WebObjs rootwebview.cpp rootwebpage.cpp rooturlschemehandler.cpp rootqt5.cpp)

ROOT_LINKER_LIBRARY(ROOTQt5WebDisplay $<TARGET_OBJECTS:RootQt5WebObjs>
                    LIBRARIES Qt5::WebEngine Qt5::WebEngineWidgets
                    DEPENDENCIES Core RHTTP ROOTWebDisplay)
