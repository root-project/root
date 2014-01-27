############################################################################
# CMakeLists.txt file for building ROOT graf2d/cocoa package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/quartz)
ROOT_USE_PACKAGE(gui/gui)

add_definitions("-ObjC++")

include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__Cocoa T*.h MODULE GCocoa LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(GCocoa *.mm  G__Cocoa.cxx LIBRARIES  "-framework Cocoa" "-framework OpenGL" ${FREETYPE_LIBRARIES} DEPENDENCIES Gui GQuartz )
ROOT_INSTALL_HEADERS()

