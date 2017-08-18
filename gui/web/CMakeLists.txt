############################################################################
# CMakeLists.txt file for building ROOT gui/qtroot package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(headers inc/*.h)
ROOT_GLOB_SOURCES(sources src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__WebGui ${headers} MODULE WebGui LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(WebGui ${sources} G__WebGui.cxx LIBRARIES Core DEPENDENCIES Gui RHTTP)

ROOT_INSTALL_HEADERS()
