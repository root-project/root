############################################################################
# CMakeLists.txt file for building ROOT proof/proof package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(core/thread)
include_directories(${CMAKE_SOURCE_DIR}/proof/proofplayer/inc)  # To avoid circular dependencies :-(

#--- if gcc 4.5.2 on Linux (Debian) use -O instead of -O2 (compiler bug)
if(ROOT_PLATFORM MATCHES "linux" AND CMAKE_COMPILER_IS_GNUCXX AND 
   GCC_VERSION VERSION_GREATER 4.5 AND GCC_VERSION VERSION_LESS 4.6)
  set_source_files_properties(src/TDataSetManager.cxx src/TDSet PROPERTIES COMPILE_FLAGS "-O")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Proof DEPENDENCIES Net Tree Thread RIO MathCore)

