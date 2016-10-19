set (Libraries  Core Hist Gpad MathCore)
set (testname stressVdt)

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND CLANG_MAJOR GREATER 3 AND CLANG_MINOR GREATER 8 )
  set (ModernClang True)
endif()

set(additional_compile_flags "-D__extern_always_inline=inline -O3")

if( ModernClang )
  set(additional_compile_flags "${additional_compile_flags} -funsafe-math-optimizations -fno-trapping-math -ffinite-math-only")
else()
  set(additional_compile_flags "${additional_compile_flags} -ffast-math")
endif()

ROOT_EXECUTABLE(${testname} ${testname}.cxx LIBRARIES ${Libraries} ADDITIONAL_COMPILE_FLAGS ${additional_compile_flags} )

ROOT_ADD_TEST(${testname} COMMAND ${testname} FAILREGEX "too") # for "too slow" and "too inaccurate" :)
