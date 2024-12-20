cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(qtweb)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

if(NOT withqt)
   if(ROOT_qt6web_FOUND)
      set(withqt 6)
   elseif(ROOT_qt5web_FOUND)
      set(withqt 5)
   else()
      error("Neither qt5web nor qt6web components were found")
   endif()
endif()

if(withqt EQUAL 6)
   find_package(Qt6 COMPONENTS Core WebEngineCore WebEngineWidgets CONFIG)
   set(qt_libs Qt6::Core Qt6::Widgets Qt6::WebEngineCore Qt6::WebEngineWidgets)
   set(root_qtlib ROOT::ROOTQt6WebDisplay)
elseif(withqt EQUAL 5)
   find_package(Qt5 COMPONENTS Widgets REQUIRED WebEngine WebEngineWidgets)
   set(qt_libs Qt5::Core Qt5::Widgets Qt5::WebEngine Qt5::WebEngineWidgets)
   set(root_qtlib ROOT::ROOTQt5WebDisplay)
else()
   error("No matching withqt parameter found, check if qt5web or qt6web components are there")
endif()

add_executable(qtweb
    ExampleWidget.ui
    ExampleWidget.cpp
    RCanvasWidget.cpp
    RGeomViewerWidget.cpp
    TCanvasWidget.cpp
    ExampleMain.cpp
)

target_link_libraries(qtweb
   ${qt_libs}
   ${ROOT_LIBRARIES} ROOT::ROOTWebDisplay ${root_qtlib} ROOT::Gpad ROOT::WebGui6 ROOT::ROOTGpadv7 ROOT::ROOTCanvasPainter ROOT::Geom ROOT::ROOTBrowserv7 ROOT::ROOTGeomViewer
)
