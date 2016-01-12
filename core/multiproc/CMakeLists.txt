############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

set(headers TMPClient.h MPSendRecv.h TProcPool.h TMPWorker.h TPoolWorker.h TPoolProcessor.h MPCode.h PoolUtils.h)

set(sources TMPClient.cxx MPSendRecv.cxx TProcPool.cxx TMPWorker.cxx TPoolWorker.cxx TPoolProcessor.cxx PoolUtils.cxx)

ROOT_GENERATE_DICTIONARY(G__MultiProc ${headers} MODULE MultiProc LINKDEF LinkDef.h)

ROOT_OBJECT_LIBRARY(MultiProcObjs ${sources} G__MultiProc.cxx)
ROOT_LINKER_LIBRARY(MultiProc $<TARGET_OBJECTS:MultiProcObjs> DEPENDENCIES Core Net TreePlayer)
ROOT_INSTALL_HEADERS(${installoptions})
