############################################################################
# CMakeLists.txt file for building ROOT graf2d/quartz package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++")

ROOT_STANDARD_LIBRARY_PACKAGE(GQuartz
                              NO_HEADERS
                              SOURCES *.mm
                              DEPENDENCIES Core
                              LIBRARIES "-framework Cocoa")
