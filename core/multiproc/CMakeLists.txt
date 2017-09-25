############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

set(headers TMPClient.h MPSendRecv.h ROOT/TProcessExecutor.hxx TProcPool.h TMPWorker.h TMPWorkerExecutor.h MPCode.h PoolUtils.h)

set(sources TMPClient.cxx MPSendRecv.cxx TProcessExecutor.cxx TMPWorker.cxx)

ROOT_STANDARD_LIBRARY_PACKAGE(MultiProc
                              OBJECT_LIBRARY
                              HEADERS ${headers}
                              LIBRARIES Core Net dl
                              DEPENDENCIES Core Net Tree)
