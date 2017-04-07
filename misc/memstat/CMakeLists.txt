############################################################################
# CMakeLists.txt file for building ROOT misc/memstat package
# @author Pere Mato, CERN
############################################################################

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set(sources TMemStat.cxx TMemStatMng.cxx TMemStatBacktrace.cxx TMemStatHelpers.cxx TMemStatHook.cxx)
set(headers TMemStatHelpers.h TMemStat.h TMemStatBacktrace.h TMemStatDef.h TMemStatMng.h TMemStatHook.h )

ROOT_GENERATE_DICTIONARY(G__MemStat ${headers} MODULE MemStat LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(MemStat ${sources} G__MemStat.cxx LIBRARIES Core ${CMAKE_DL_LIBS} DEPENDENCIES Tree Gpad Graf)

ROOT_INSTALL_HEADERS()

