############################################################################
# CMakeLists.txt file for building ROOT core/thread package
############################################################################
ROOT_USE_PACKAGE(core)

set(headers TCondition.h TConditionImp.h TMutex.h TMutexImp.h
            TRWLock.h TSemaphore.h TThread.h TThreadFactory.h
            TThreadImp.h TAtomicCount.h TThreadPool.h ThreadLocalStorage.h)
if(NOT WIN32)
  set(headers ${headers} TPosixCondition.h TPosixMutex.h
                         TPosixThread.h TPosixThreadFactory.h PosixThreadInc.h)
  set(installoptions OPTIONS REGEX "Win32" EXCLUDE)
else()
  set(headers ${headers} TWin32Condition.h TWin32Mutex.h
                         TWin32Thread.h TWin32ThreadFactory.h)
  set(installoptions OPTIONS REGEX "Posix" EXCLUDE)
endif()

set(sources TCondition.cxx TConditionImp.cxx TMutex.cxx TMutexImp.cxx
            TRWLock.cxx TSemaphore.cxx TThread.cxx TThreadFactory.cxx
            TThreadImp.cxx)
if(NOT WIN32)
  set(sources ${sources} TPosixCondition.cxx TPosixMutex.cxx
                         TPosixThread.cxx TPosixThreadFactory.cxx)
else()
  set(sources ${sources} TWin32Condition.cxx TWin32Mutex.cxx
                         TWin32Thread.cxx TWin32ThreadFactory.cxx)
endif()

ROOT_GENERATE_DICTIONARY(G__Thread ${headers} STAGE1 MODULE Thread LINKDEF LinkDef.h)

ROOT_OBJECT_LIBRARY(ThreadObjs ${sources} G__Thread.cxx)
ROOT_LINKER_LIBRARY(Thread $<TARGET_OBJECTS:ThreadObjs> LIBRARIES  ${CMAKE_THREAD_LIBS_INIT} DEPENDENCIES Core)
ROOT_INSTALL_HEADERS(${installoptions})
