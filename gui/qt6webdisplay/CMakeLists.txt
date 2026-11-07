# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/qt6webdisplay package
############################################################################

find_package(Qt6 COMPONENTS Core WebEngineCore WebEngineWidgets CONFIG)

if(NOT Qt6_FOUND)
  message(SEND_ERROR "Could NOT find Qt6 (WebEngineCore, WebEngineWidgets), install missing packages on the system or disable option 'qt6web'")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

ROOT_LINKER_LIBRARY(ROOTQt6WebDisplay
     rootwebview.cpp
     rootwebpage.cpp
     rooturlschemehandler.cpp
     rootqt6.cpp
  LIBRARIES
    Qt6::WebEngineCore
    Qt6::WebEngineWidgets
  DEPENDENCIES
    Core
    RHTTP
    ROOTWebDisplay
)
