set (Libraries  Core Hist Gpad MathCore)
set (testname stressVdt)


if( CMAKE_COMPILER_IS_GNUCXX AND GCC_MAJOR GREATER 3 AND GCC_MINOR GREATER 6 )
  set (ModernGcc True)
endif()

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND CLANG_MAJOR GREATER 2 AND CLANG_MINOR GREATER 3 )
  set (ModernClang True)
  # __extern_always_inline is not understood by clang as of version 3.5
  set (alwaysInlineClang "-D__extern_always_inline=inline")
endif()

if( ModernGcc OR ModernClang)
   set(additional_compile_flags "-O3 -ffast-math ${alwaysInlineClang} ")
endif()

ROOT_EXECUTABLE(${testname} ${testname}.cxx LIBRARIES ${Libraries} ADDITIONAL_COMPILE_FLAGS ${additional_compile_flags} )

ROOT_ADD_TEST(${testname} COMMAND ${testname} FAILREGEX "too") # for "too slow" and "too inaccurate" :)
