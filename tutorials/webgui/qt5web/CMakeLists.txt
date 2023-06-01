cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(qt5web)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

if(with_qt6)
   find_package(Qt6 COMPONENTS Core WebEngineCore WebEngineWidgets CONFIG)
   set(qt_libs Qt6::Core Qt6::Widgets Qt6::WebEngineCore Qt6::WebEngineWidgets)
   set(root_qtlib ROOT::ROOTQt6WebDisplay)
   set(appname qt6web)
else()
   find_package(Qt5 COMPONENTS Widgets REQUIRED WebEngine WebEngineWidgets)
   set(qt_libs Qt5::Core Qt5::Widgets Qt5::WebEngine Qt5::WebEngineWidgets)
   set(root_qtlib ROOT::ROOTQt5WebDisplay)
   set(appname qt5web)
endif()

add_executable(${appname}
    ExampleWidget.ui
    ExampleWidget.cpp
    RCanvasWidget.cpp
    RGeomViewerWidget.cpp
    TCanvasWidget.cpp
    ExampleMain.cpp
)

target_link_libraries(${appname}
   ${qt_libs}
   ${ROOT_LIBRARIES} ROOT::ROOTWebDisplay ${root_qtlib} ROOT::Gpad ROOT::WebGui6 ROOT::ROOTGpadv7 ROOT::ROOTCanvasPainter ROOT::Geom ROOT::ROOTBrowserv7 ROOT::ROOTGeomViewer
)
