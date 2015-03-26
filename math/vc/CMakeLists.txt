set(Vc_INSIDE_ROOT TRUE)

if (CMAKE_BUILD_TYPE STREQUAL Optimized)
  set(VC_EXTRA_CONSERVATIVE_FLAGS "-fno-fast-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VC_EXTRA_CONSERVATIVE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VC_EXTRA_CONSERVATIVE_FLAGS}")
endif()

include(cmake/VcMacros.cmake)
include(cmake/AddTargetProperty.cmake)
vc_determine_compiler()
vc_set_preferred_compiler_flags(WARNING_FLAGS BUILDTYPE_FLAGS)
add_definitions("${Vc_DEFINITIONS}")
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(libvc_compile_flags "-DVC_COMPILE_LIB")
vc_compile_for_all_implementations(_objs src/trigonometric.cpp FLAGS ${libvc_compile_flags}
      ONLY SSE2 SSE3 SSSE3 SSE4_1 AVX SSE+XOP+FMA4 AVX+XOP+FMA4)
set(_srcs src/const.cpp src/cpuid.cpp src/support.cpp ${_objs})
if(USE_AVX)
   list(APPEND _srcs src/avx_sorthelper.cpp)
else()
   if(NOT Vc_AVX_INTRINSICS_BROKEN)
      # we'd still like to have avx_sorthelper.cpp built in, but that requires compilation with -mavx (or a comparable flag)
      foreach(_flag "-xAVX" "-mavx" "/arch:AVX")
         AddCompilerFlag("${_flag}" CXX_RESULT _flag_works)
         if(_flag_works)
            if(_flag STREQUAL "-xAVX")
               set(_flag "${_flag} -diag-disable 10121") # disable the warning "overriding -xSSE4.2 with -xAVX"
            endif()
            list(APPEND _srcs src/avx_sorthelper.cpp)
            set_source_files_properties(src/avx_sorthelper.cpp PROPERTIES COMPILE_FLAGS "${_flag}")
            break()
         endif()
      endforeach()
   endif()
endif()
add_library(Vc STATIC ${_srcs})
add_target_property(Vc COMPILE_FLAGS ${libvc_compile_flags})

set_property(GLOBAL APPEND PROPERTY ROOT_EXPORTED_TARGETS Vc)

if(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fp-model source")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model source")
endif()

install(TARGETS Vc RUNTIME DESTINATION bin
                   LIBRARY DESTINATION lib
                   ARCHIVE DESTINATION lib
                   COMPONENT libraries)
ROOT_INSTALL_HEADERS(include/)

ROOT_BUILD_OPTION(vc_examples OFF "Build Vc examples")
ROOT_BUILD_OPTION(vc_tests OFF "Build Vc unit tests")
if(vc_examples OR vc_tests)
   add_custom_target(other VERBATIM)
   add_custom_target(Scalar COMMENT "build Vc Scalar code" VERBATIM)
   add_custom_target(SSE COMMENT "build Vc SSE code" VERBATIM)
   add_custom_target(AVX COMMENT "build Vc AVX code" VERBATIM)
endif()
if(vc_examples)
   add_subdirectory(examples)
endif()
if(vc_tests)
   add_subdirectory(tests)
endif()
