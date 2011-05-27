############################################################################
# CMakeLists.txt file for building ROOT graf3d/x3d package
############################################################################

set(libname X3d)

ROOT_USE_PACKAGE(graf3d/g3d)
ROOT_USE_PACKAGE(gui/gui)

ROOT_GENERATE_DICTIONARY(G__X3D TViewerX3D.h TX3DFrame.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Graf3d Gui)
ROOT_LINKER_LIBRARY(${libname} *.cxx G__X3D.cxx LIBRARIES Graf3d Gui)
ROOT_INSTALL_HEADERS()
