if(( CMAKE_COMPILER_IS_GNUCXX AND GCC_MAJOR GREATER 3 AND GCC_MINOR GREATER 6 ) OR
   ( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND CLANG_MAJOR GREATER 2 AND CLANG_MINOR GREATER 3 )  ) 
   message(STATUS "Detected a modern gcc/clang compiler.")
   AddCompilerFlag("-O3 -ffast-math")
endif()

set (Libraries  Core Hist Gpad MathCore)
set (testname stressVdt) 

ROOT_EXECUTABLE(${testname} ${testname}.cxx LIBRARIES ${Libraries})
ROOT_ADD_TEST(${testname} COMMAND ${testname})