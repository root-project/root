############################################################################
# CMakeLists.txt file for building ROOT misc/memstat package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(tree/tree)

set(sources TMemStat.cxx TMemStatMng.cxx TMemStatBacktrace.cxx TMemStatHelpers.cxx TMemStatHook.cxx)
set(headers TMemStatHelpers.h TMemStat.h TMemStatBacktrace.h TMemStatDef.h TMemStatMng.h TMemStatHook.h )

ROOT_GENERATE_DICTIONARY(G__MemStat ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(MemStat LINKDEF LinkDef.h DEPENDENCIES Tree Gpad Graf)
ROOT_LINKER_LIBRARY(MemStat ${sources} G__MemStat.cxx  LIBRARIES)

ROOT_INSTALL_HEADERS()

