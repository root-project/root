############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Netx
  HEADERS
    TXNetFile.h
    TXNetFileStager.h
    TXNetSystem.h
  SOURCES
    src/TXNetFile.cxx
    src/TXNetFileStager.cxx
    src/TXNetSystem.cxx
  DEPENDENCIES
    Net
    RIO
    Thread
  BUILTINS
    XROOTD
)

target_compile_options(Netx PRIVATE ${XROOTD_CFLAGS})
target_include_directories(Netx PRIVATE ${XROOTD_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/proof/proofd/inc)
target_link_libraries(Netx PRIVATE ${XROOTD_LIBRARIES})
