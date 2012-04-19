############################################################################
# CMakeLists.txt file for building ROOT graf2d/cocoa package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/quartz)
ROOT_USE_PACKAGE(gui/gui)

add_definitions("-ObjC++ -std=c++11")

ROOT_GENERATE_DICTIONARY(G__Cocoa T*.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(GCocoa LINKDEF LinkDef.h DEPENDENCIES Gui GQuartz )
ROOT_LINKER_LIBRARY(GCocoa *.mm  G__Cocoa.cxx LIBRARIES  "-framework Cocoa" DEPENDENCIES Gui GQuartz )
ROOT_INSTALL_HEADERS()

