############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

set(HEADERS
  ROOT/TAdoptAllocator.hxx
  ROOT/TVec.hxx
)

set(SOURCES
  TAdoptAllocator.cxx
  TVec.cxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(VecOps
                              HEADERS ${HEADERS}
                              SOURCES ${SOURCES}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core)

ROOT_ADD_TEST_SUBDIRECTORY(test)
