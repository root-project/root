############################################################################
# CMakeLists.txt file for building ROOT math/vecops package
############################################################################

if(builtin_vdt)
  link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  include_directories(${HEADER_OUTPUT_PATH})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTVecOps
  HEADERS
    ROOT/RAdoptAllocator.hxx
    ROOT/RVec.hxx
  SOURCES
    src/RAdoptAllocator.cxx
    src/RVec.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
    -I${VDT_INCLUDE_DIR}
  DEPENDENCIES
    Core
)

if(builtin_vdt)
  target_link_libraries(ROOTVecOps PUBLIC $<INSTALL_INTERFACE:vdt>)
  target_include_directories(ROOTVecOps PUBLIC $<INSTALL_INTERFACE:include>)
elseif(vdt)
  target_link_libraries(ROOTVecOps PUBLIC ${VDT_LIBRARIES})
  target_include_directories(ROOTVecOps PUBLIC ${VDT_INCLUDE_DIRS})
endif()

if(MSVC)
  target_compile_options(ROOTVecOps PRIVATE -O2 /fp:fast)
else()
  target_compile_options(ROOTVecOps PRIVATE -O3 -ffast-math)
endif()

include(CheckCXXSymbolExists)
check_symbol_exists(m __sqrt_finite HAVE_FINITE_MATH)
if(NOT HAVE_FINITE_MATH AND NOT MSVC)
  target_compile_options(ROOTVecOps PRIVATE -fno-finite-math-only)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
