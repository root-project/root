############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

set(HEADERS
  ROOT/RAdoptAllocator.hxx
  ROOT/RVec.hxx
)

set(SOURCES
  src/RAdoptAllocator.cxx
  src/RVec.cxx
)

if(vdt)
  # needed for rootcling, to be removed
  include_directories(${VDT_INCLUDE_DIRS})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTVecOps
                              HEADERS ${HEADERS}
                              SOURCES ${SOURCES}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES Core)

if(vdt)
  target_link_libraries(ROOTVecOps PUBLIC Vdt::Vdt)
endif()

target_compile_options(ROOTVecOps PRIVATE -O3 -ffast-math)

include(CheckCXXSymbolExists)
check_symbol_exists(m __sqrt_finite HAVE_FINITE_MATH)
if(NOT HAVE_FINITE_MATH)
  target_compile_options(ROOTVecOps PRIVATE -fno-finite-math-only)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
