############################################################################
# CMakeLists.txt file for building ROOT core/thread package
############################################################################

set(headers TAtomicCount.h TCondition.h TConditionImp.h TMutex.h TMutexImp.h
            TRWLock.h ROOT/TRWSpinLock.hxx TSemaphore.h TThread.h TThreadFactory.h
            TThreadImp.h ROOT/TThreadedObject.hxx TThreadPool.h
            ThreadLocalStorage.h ROOT/TSpinMutex.hxx)
if(NOT WIN32)
  set(headers ${headers} TPosixCondition.h TPosixMutex.h
                         TPosixThread.h TPosixThreadFactory.h PosixThreadInc.h)
  set(installoptions FILTER "Win32")
else()
  set(headers ${headers} TWin32Condition.h TWin32Mutex.h
                         TWin32Thread.h TWin32ThreadFactory.h)
  set(installoptions FILTER "Posix")
endif()

set(sources TCondition.cxx TConditionImp.cxx TMutex.cxx TMutexImp.cxx
            TRWLock.cxx TRWSpinLock.cxx TSemaphore.cxx TThread.cxx TThreadFactory.cxx
            TThreadImp.cxx)
if(NOT WIN32)
  set(sources ${sources} TPosixCondition.cxx TPosixMutex.cxx
                         TPosixThread.cxx TPosixThreadFactory.cxx)
else()
  set(sources ${sources} TWin32Condition.cxx TWin32Mutex.cxx
                         TWin32Thread.cxx TWin32ThreadFactory.cxx)
endif()

if (imt)
  set(sources_imt TImplicitMT.cxx TThreadExecutor.cxx)
  set(headers_imt ROOT/TThreadExecutor.hxx)
endif()

include_directories(${TBB_INCLUDE_DIRS})

ROOT_GENERATE_DICTIONARY(G__Thread ${headers} ${headers_imt} STAGE1 MODULE Thread LINKDEF LinkDef.h)

ROOT_OBJECT_LIBRARY(ThreadObjs ${sources} G__Thread.cxx)
ROOT_LINKER_LIBRARY(Thread $<TARGET_OBJECTS:ThreadObjs> ${sources_imt} LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${TBB_LIBRARIES} DEPENDENCIES Core BUILTINS TBB)
ROOT_INSTALL_HEADERS(${installoptions})
