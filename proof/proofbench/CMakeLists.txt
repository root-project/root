############################################################################
# CMakeLists.txt file for building ROOT proof/proofbench package
############################################################################

ROOT_USE_PACKAGE(proof/proof)
ROOT_USE_PACKAGE(proof/proofplayer)

ROOT_GLOB_HEADERS(headers inc/TProof*.h)
ROOT_GLOB_SOURCES(sources src/TProof*.cxx)

ROOT_GENERATE_DICTIONARY(G__ProofBench ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(ProofBench LINKDEF LinkDef.h DEPENDENCIES Net RIO)
ROOT_LINKER_LIBRARY(ProofBench ${sources} G__ProofBench.cxx LIBRARIES Core  DEPENDENCIES Hist Gpad ProofPlayer)
ROOT_INSTALL_HEADERS()
