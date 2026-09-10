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
    ROOT/TWin32Condition.h
    ROOT/TWin32Mutex.h
    ROOT/TWin32Thread.h
    ROOT/TWin32ThreadFactory.h
  )
else()
  set(PLATFORM_FILTER FILTER "Win32")
  set(PLATFORM_HEADERS
    ROOT/TPosixCondition.h
    ROOT/TPosixMutex.h
    ROOT/TPosixThread.h
    ROOT/TPosixThreadFactory.h
    PosixThreadInc.h
  )
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Thread
  HEADERS
    ${PLATFORM_HEADERS}
    ROOT/TAtomicCount.h
    ROOT/TCondition.h
    ROOT/TConditionImp.h
    ROOT/TMutex.h
    ROOT/TMutexImp.h
    ROOT/TRWLock.h
    ROOT/TSemaphore.h
    ROOT/TThreadFactory.h
    ROOT/TThread.h
    ROOT/TThreadImp.h
    ROOT/TThreadPool.h
    ROOT/TRWSpinLock.hxx
    ROOT/TSpinMutex.hxx
    ROOT/TThreadedObject.hxx
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
  INSTALL_OPTIONS ${installoptions}
)

target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT> # bw compatibility
)

target_link_libraries(Thread PUBLIC ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(Thread PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT> # bw compatibility
)

if(TARGET TBB::tbb)
  target_link_libraries(Thread PRIVATE TBB::tbb)
  target_compile_definitions(Thread PUBLIC ROOT_CORE_THREAD_TBB)
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
