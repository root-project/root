cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(qtweb)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

find_package(Qt6 COMPONENTS Core WebEngineCore WebEngineWidgets CONFIG)

add_executable(qtweb
    ExampleWidget.ui
    ExampleWidget.cpp
    RCanvasWidget.cpp
    RGeomViewerWidget.cpp
    TCanvasWidget.cpp
    ExampleMain.cpp
)

target_link_libraries(qtweb
   Qt6::Core Qt6::Widgets Qt6::WebEngineCore Qt6::WebEngineWidgets
   ${ROOT_LIBRARIES} ROOT::ROOTWebDisplay ROOT::ROOTQt6WebDisplay ROOT::Gpad ROOT::WebGui6 ROOT::ROOTGpadv7 ROOT::ROOTCanvasPainter ROOT::Geom ROOT::ROOTBrowserv7 ROOT::ROOTGeomViewer
)
