############################################################################
# CMakeLists.txt file for building ROOT core/imt package
############################################################################

set(sources base.cxx)

if (imt)
  set(headers ROOT/TPoolManager.hxx ROOT/TThreadExecutor.hxx)
  ROOT_GENERATE_DICTIONARY(G__Imt ${headers} STAGE1 MODULE Imt LINKDEF LinkDef.h  DEPENDENCIES Core Thread BUILTINS TBB) # For auto{loading,parsing}
  set(sources ${sources} TImplicitMT.cxx TThreadExecutor.cxx TPoolManager.cxx G__Imt.cxx)
endif()

include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

ROOT_LINKER_LIBRARY(Imt ${sources} LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES} DEPENDENCIES Core Thread BUILTINS TBB)
ROOT_INSTALL_HEADERS(${installoptions})
