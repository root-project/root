############################################################################
# CMakeLists.txt file for building ROOT core/imt package
############################################################################

set(sources base.cxx)

if (imt)
  set(sources ${sources} TImplicitMT.cxx TThreadExecutor.cxx TPoolManager.cxx)
endif()

include_directories(${TBB_INCLUDE_DIRS})

ROOT_OBJECT_LIBRARY(ImtObjs ${sources})
ROOT_LINKER_LIBRARY(Imt $<TARGET_OBJECTS:ImtObjs> LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES} DEPENDENCIES Core Thread BUILTINS TBB)
ROOT_INSTALL_HEADERS(${installoptions})
