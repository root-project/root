cmake_minimum_required(VERSION 3.16)
project(qt6canv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

qt_add_executable(qt6canv
    main.cpp
    mainwidget.h
    mainwidget.cpp
)

target_link_libraries(qt6canv PRIVATE Qt6::Widgets ${ROOT_LIBRARIES} ROOT::Gpad ROOT::ROOTQt6Canvas)
