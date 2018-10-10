############################################################################
# CMakeLists.txt file for building ROOT gui/web package
############################################################################

ROOT_GLOB_HEADERS(headers inc/*.h)
ROOT_GLOB_SOURCES(sources src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__WebGui ${headers} MODULE WebGui LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(WebGui ${sources} G__WebGui.cxx LIBRARIES Core DEPENDENCIES Gui ROOTWebDisplay)

ROOT_INSTALL_HEADERS()
