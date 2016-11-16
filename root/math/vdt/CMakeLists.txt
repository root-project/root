if (ROOT_vdt_FOUND AND NOT ROOT_CLASSIC_BUILD)

set (testname stressVdt)

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # __extern_always_inline is not understood by clang as of version 3.5
  set (alwaysInlineClang "-D__extern_always_inline=inline")
endif()

if( CMAKE_COMPILER_IS_GNUCXX )
   set(additional_compile_flags "-O3 -ffast-math")
elseif(alwaysInlineClang)
   set(additional_compile_flags "-O3 -funsafe-math-optimizations -fno-trapping-math -ffinite-math-only ${alwaysInlineClang} ")
endif()

ROOTTEST_GENERATE_EXECUTABLE(${testname}
                             ${testname}.cxx
                             ADDITIONAL_COMPILE_FLAGS ${additional_compile_flags}
                             LIBRARIES Core Hist Gpad MathCore)

ROOTTEST_ADD_TEST(${testname}
                  EXEC ${testname}
                  FAILREGEX "too" # for "too slow" and "too inaccurate" :)
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})
endif()
