############################################################################
# CMakeLists.txt file for building ROOT proof/proofplayer package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(headers inc/*.h)
list(REMOVE_ITEM headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/TProofDraw.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h
                         ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDefDraw.h)

ROOT_GLOB_SOURCES(sources src/*.cxx)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TProofDraw.cxx)

ROOT_STANDARD_LIBRARY_PACKAGE(ProofPlayer
                              NO_INSTALL_HEADERS
                              HEADERS ${headers}
                              SOURCES ${sources}
                              DEPENDENCIES Proof Hist RIO Tree Net Thread MathCore)

ROOT_STANDARD_LIBRARY_PACKAGE(ProofDraw
                              NO_INSTALL_HEADERS
                              HEADERS TProofDraw.h
                              LINKDEF LinkDefDraw.h
                              SOURCES TProofDraw.cxx
                              DEPENDENCIES ProofPlayer TreePlayer)

ROOT_INSTALL_HEADERS()
