############################################################################
# CMakeLists.txt file for building ROOT graf2d/cocoa package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++")

if (cxxmodules)
  # FIXME: Disable modules for ObjC/ObjC++. Modules for cocoa and quarts have to
  # be treated in a special way. Eg. it seems that they cannot be nested in the
  # ROOT top-most module, because an implicit build of this module in ObjC TU
  # causes issues in TBonjourBrowser (when incluing dns_sd.h) on OSX.
  string(REPLACE "${ROOT_CXXMODULES_CXXFLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "${ROOT_CXXMODULES_CFLAGS}" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif(cxxmodules)

include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__GCocoa T*.h MODULE GCocoa LINKDEF LinkDef.h DEPENDENCIES Gui GQuartz)

ROOT_LINKER_LIBRARY(GCocoa *.mm  G__GCocoa.cxx LIBRARIES  "-framework Cocoa" "-framework OpenGL" ${FREETYPE_LIBRARIES} DEPENDENCIES Gui GQuartz )
ROOT_INSTALL_HEADERS()

