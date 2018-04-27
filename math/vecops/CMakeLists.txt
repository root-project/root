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

if(vdt)
  # needed for rootcling, to be removed
  include_directories(${VDT_INCLUDE_DIRS})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(VecOps
                              HEADERS ${HEADERS}
                              SOURCES ${SOURCES}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core)

if(vdt)
  target_link_libraries(VecOps Vdt::Vdt)
endif()

target_compile_options(VecOps PRIVATE -Ofast)

ROOT_ADD_TEST_SUBDIRECTORY(test)
