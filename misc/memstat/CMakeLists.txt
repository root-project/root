############################################################################
# CMakeLists.txt file for building ROOT misc/memstat package
# @author Pere Mato, CERN
############################################################################

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set(sources TMemStat.cxx TMemStatMng.cxx TMemStatBacktrace.cxx TMemStatHelpers.cxx TMemStatHook.cxx)
set(headers TMemStatHelpers.h TMemStat.h TMemStatBacktrace.h TMemStatDef.h TMemStatMng.h TMemStatHook.h )

ROOT_STANDARD_LIBRARY_PACKAGE(MemStat
                              HEADERS ${headers}
                              SOURCES ${sources}
                              LIBRARIES ${CMAKE_DL_LIBS}
                              DEPENDENCIES Tree Gpad Graf)
