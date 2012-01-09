############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################
ROOT_USE_PACKAGE(core)

ROOT_GENERATE_DICTIONARY(G__Rint *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Rint LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(Rint *.cxx G__Rint.cxx DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()
