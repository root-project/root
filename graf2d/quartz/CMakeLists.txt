############################################################################
# CMakeLists.txt file for building ROOT graf2d/quartz package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++")

ROOT_GENERATE_DICTIONARY(G__Quartz "" MODULE GQuartz LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(GQuartz *.mm  G__Quartz.cxx LIBRARIES  "-framework Cocoa" DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()

