# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/thread package
############################################################################

if(WIN32)
  set(PLATFORM_FILTER FILTER "Posix")
  set(PLATFORM_HEADERS
    TWin32Condition.h
    TWin32Mutex.h
    TWin32Thread.h
    TWin32ThreadFactory.h
  )
else()
  set(PLATFORM_FILTER FILTER "Win32")
  set(PLATFORM_HEADERS
    TPosixCondition.h
    TPosixMutex.h
    TPosixThread.h
    TPosixThreadFactory.h
    PosixThreadInc.h
  )
endif()

set(THREAD_HEADERS
  TAtomicCount.h
  TCondition.h
  TConditionImp.h
  TMutex.h
  TMutexImp.h
  TRWLock.h
  TSemaphore.h
  TThreadFactory.h
  TThread.h
  TThreadImp.h
  TThreadPool.h
  ROOT/TRWSpinLock.hxx
  ROOT/TSpinMutex.hxx
  ROOT/TThreadedObject.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(Thread
  HEADERS
    ${PLATFORM_HEADERS}
    ${THREAD_HEADERS}
  SOURCES
    src/TCondition.cxx
    src/TConditionImp.cxx
    src/TMutex.cxx
    src/TMutexImp.cxx
    src/TReentrantRWLock.cxx
    src/TRWLock.cxx
    src/TRWMutexImp.cxx
    src/TRWSpinLock.cxx
    src/TSemaphore.cxx
    src/TThread.cxx
    src/TThreadFactory.cxx
    src/TThreadImp.cxx
  STAGE1
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
  BUILTINS
    TBB
  INSTALL_OPTIONS ${installoptions}
)

target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

set(RELATIVE_INC_HEADERS ${PLATFORM_HEADERS} ${THREAD_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Thread
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          ${RELATIVE_INC_HEADERS}
          inc/LinkDef.h
  )
endif()

target_link_libraries(Thread PUBLIC ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(Thread PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

if((tbb OR builtin_tbb) AND NOT MSVC)
  target_include_directories(Thread PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(Thread PRIVATE ${TBB_LIBRARIES})
  set_target_properties(Thread PROPERTIES COMPILE_FLAGS "${TBB_CXXFLAGS}")
endif()

if(WIN32)
  target_sources(Thread PRIVATE
    src/TWin32Condition.cxx
    src/TWin32Mutex.cxx
    src/TWin32Thread.cxx
    src/TWin32ThreadFactory.cxx
  )
  target_include_directories(Thread PRIVATE ${CMAKE_SOURCE_DIR}/core/winnt/inc)
else()
  target_sources(Thread PRIVATE
    src/TPosixCondition.cxx
    src/TPosixMutex.cxx
    src/TPosixThread.cxx
    src/TPosixThreadFactory.cxx
  )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
