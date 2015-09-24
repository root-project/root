############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

set(headers TMPClient.h MPSendRecv.h TPool.h TMPWorker.h TPoolWorker.h MPCode.h PoolCode.h)

set(sources TMPClient.cxx MPSendRecv.cxx TPool.cxx TMPWorker.cxx TPoolWorker.cxx)

ROOT_GENERATE_DICTIONARY(G__MultiProc ${headers} MODULE MultiProc LINKDEF LinkDef.h)

ROOT_OBJECT_LIBRARY(MultiProcObjs ${sources} G__MultiProc.cxx)
ROOT_LINKER_LIBRARY(MultiProc $<TARGET_OBJECTS:MultiProcObjs> DEPENDENCIES Core Net)
ROOT_INSTALL_HEADERS(${installoptions})
