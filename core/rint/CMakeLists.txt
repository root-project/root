############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################

ROOT_GENERATE_DICTIONARY(G__Rint *.h STAGE1 MODULE Rint LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Core)

ROOT_LINKER_LIBRARY(Rint *.cxx G__Rint.cxx DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()
