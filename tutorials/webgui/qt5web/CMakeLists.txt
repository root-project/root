cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(qt5web)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED WebEngine WebEngineWidgets)

add_executable(qt5web
    ExampleWidget.ui
    ExampleWidget.cpp
    RCanvasWidget.cpp
    RGeomViewerWidget.cpp
    TCanvasWidget.cpp
    ExampleMain.cpp
)

target_include_directories(qt5web PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(qt5web
   Qt5::Core Qt5::Widgets Qt5::WebEngine Qt5::WebEngineWidgets
   ${ROOT_LIBRARIES} ROOT::ROOTWebDisplay ROOT::ROOTQt5WebDisplay ROOT::Gpad ROOT::WebGui6 ROOT::ROOTGpadv7 ROOT::ROOTCanvasPainter ROOT::Geom ROOT::ROOTBrowserv7 ROOT::ROOTEve
)
