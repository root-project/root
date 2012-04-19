############################################################################
# CMakeLists.txt file for building ROOT graf2d/quartz package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++ -std=c++11")

ROOT_GENERATE_DICTIONARY(G__Quartz "" LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(GQuartz LINKDEF LinkDef.h DEPENDENCIES )
ROOT_LINKER_LIBRARY(GQuartz *.mm  G__Quartz.cxx LIBRARIES  "-framework Cocoa" DEPENDENCIES )
ROOT_INSTALL_HEADERS()

