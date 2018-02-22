############################################################################
# CMakeLists.txt file for building ROOT gui/qt5webdisplay package
############################################################################

set(libname ROOTQt5WebDisplay)

find_package(Qt5Core)

find_package(Qt5WebEngine)

find_package(Qt5WebEngineWidgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# avoid cmake error when all qt-related options are enabled
set_property(SOURCE dummy.cxx PROPERTY SKIP_AUTOMOC ON)

include_directories(${Qt5WebEngine_INCLUDE_DIRS})
include_directories(${Qt5WebEngineWidgets_INCLUDE_DIRS})

ROOT_OBJECT_LIBRARY(RootQt5WebObjs rootwebview.cpp rootwebpage.cpp rooturlschemehandler.cpp rootqt5.cpp)

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:RootQt5WebObjs>
                    LIBRARIES Qt5::WebEngine Qt5::WebEngineWidgets
                    DEPENDENCIES Core RHTTP)
