# Library which powers fast batch computations in Roofit.

ROOT_LINKER_LIBRARY(RooBatchCompute
    src/Initialisation.cxx
  DEPENDENCIES
    Core
    MathCore
)

target_include_directories(RooBatchCompute PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/res>)

if(vdt OR builtin_vdt)
  target_link_libraries(RooBatchCompute PUBLIC VDT::VDT)
endif()
if(builtin_vdt)
  add_dependencies(RooBatchCompute VDT)
endif()


############################################################################################################################################
# Instantiations of the shared objects which provide the actual computation functions.

set(shared_object_sources src/RooBatchCompute.cxx src/ComputeFunctions.cxx)

# Generic implementation for CPUs that don't support vector instruction sets.
ROOT_LINKER_LIBRARY(RooBatchCompute_GENERIC ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
target_compile_options(RooBatchCompute_GENERIC  PRIVATE ${common-flags} -DRF_ARCH=GENERIC)

# Windows platform and ICC compiler need special code and testing, thus the feature has not been implemented yet for these.
if (ROOT_PLATFORM MATCHES "linux|macosx" AND CMAKE_SYSTEM_PROCESSOR MATCHES x86_64 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

  target_compile_options(RooBatchCompute PRIVATE -DR__RF_ARCHITECTURE_SPECIFIC_LIBS)

  ROOT_LINKER_LIBRARY(RooBatchCompute_SSE4.1  ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX     ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX2    ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)
  ROOT_LINKER_LIBRARY(RooBatchCompute_AVX512  ${shared_object_sources} TYPE SHARED DEPENDENCIES RooBatchCompute)

  # Flags -fno-signaling-nans, -fno-trapping-math and -O3 are necessary to enable autovectorization (especially for GCC).
  set(common-flags $<$<CXX_COMPILER_ID:GNU>:-fno-signaling-nans>)
  list(APPEND common-flags $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>: -fno-trapping-math -O3>)

  target_compile_options(RooBatchCompute_SSE4.1  PRIVATE ${common-flags} -msse4    -DRF_ARCH=SSE4)
  target_compile_options(RooBatchCompute_AVX     PRIVATE ${common-flags} -mavx     -DRF_ARCH=AVX)
  target_compile_options(RooBatchCompute_AVX2    PRIVATE ${common-flags} -mavx2    -DRF_ARCH=AVX2)
  target_compile_options(RooBatchCompute_AVX512  PRIVATE ${common-flags} -march=skylake-avx512 -DRF_ARCH=AVX512)

endif() # vector versions of library

if (cuda)
  set(shared_object_sources_cu src/RooBatchCompute.cu src/ComputeFunctions.cu src/CudaInterface.cu)
  ROOT_LINKER_LIBRARY(RooBatchCompute_CUDA  ${shared_object_sources_cu} TYPE SHARED DEPENDENCIES RooBatchCompute)
  target_compile_options(RooBatchCompute_CUDA  PRIVATE -lineinfo --expt-relaxed-constexpr)
endif()
