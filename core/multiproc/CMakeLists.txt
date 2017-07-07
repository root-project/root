############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

set(headers TMPClient.h MPSendRecv.h ROOT/TProcessExecutor.hxx TProcPool.h TMPWorker.h TMPWorkerExecutor.h MPCode.h PoolUtils.h)

set(sources TMPClient.cxx MPSendRecv.cxx TProcessExecutor.cxx TMPWorker.cxx)

ROOT_GENERATE_DICTIONARY(G__MultiProc ${headers} MODULE MultiProc LINKDEF LinkDef.h DEPENDENCIES Core Net Tree)

ROOT_OBJECT_LIBRARY(MultiProcObjs ${sources} G__MultiProc.cxx)
ROOT_LINKER_LIBRARY(MultiProc $<TARGET_OBJECTS:MultiProcObjs> DEPENDENCIES Core Net dl)
ROOT_INSTALL_HEADERS()
