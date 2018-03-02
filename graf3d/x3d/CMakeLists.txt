############################################################################
# CMakeLists.txt file for building ROOT graf3d/x3d package
############################################################################

include_directories(${X11_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(X3d
                              HEADERS TViewerX3D.h TX3DFrame.h
                              SOURCES *.cxx *.c
                              LIBRARIES ${X11_LIBRARIES}
                              DEPENDENCIES Graf3d Gui)
