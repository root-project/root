############################################################################
# CMakeLists.txt file for building ROOT gui/qt5webdisplay package
############################################################################

set(libname ROOTQt5WebDisplay)

if(POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()

find_package(Qt5 COMPONENTS Core WebEngine WebEngineWidgets REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${Qt5WebEngine_INCLUDE_DIRS})
include_directories(${Qt5WebEngineWidgets_INCLUDE_DIRS})

ROOT_OBJECT_LIBRARY(RootQt5WebObjs rootwebview.cpp rootwebpage.cpp rooturlschemehandler.cpp rootqt5.cpp)

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:RootQt5WebObjs>
                    LIBRARIES Qt5::WebEngine Qt5::WebEngineWidgets
                    DEPENDENCIES Core RHTTP)
