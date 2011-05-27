############################################################################
# CMakeLists.txt file for building ROOT proof/proof package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(core/thread)
include_directories(${CMAKE_SOURCE_DIR}/proof/proofplayer/inc)  # To avoid circular dependencies :-(


ROOT_STANDARD_LIBRARY_PACKAGE(Proof DEPENDENCIES Net Tree Thread RIO MathCore)

