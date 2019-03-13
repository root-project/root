############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(MultiProc
  OBJECT_LIBRARY
  HEADERS
    MPCode.h
    MPSendRecv.h
    PoolUtils.h
    TMPClient.h
    TMPWorker.h
    TMPWorkerExecutor.h
    TProcPool.h
    ROOT/TProcessExecutor.hxx
  SOURCES
    src/MPSendRecv.cxx
    src/TMPClient.cxx
    src/TMPWorker.cxx
    src/TProcessExecutor.cxx
  LIBRARIES
    ${CMAKE_DL_LIBS}
  DEPENDENCIES
    Core
    Net
)
