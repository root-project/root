GET_FILENAME_COMPONENT(_cmake_cxx_compiler_name ${CMAKE_CXX_COMPILER} NAME)

IF (CMAKE_COMPILER_IS_GNUCXX)
   # GNU C++:
   SET(REFLEX_COMPILER_CONFIG "config/compiler/GCC")
ELSEIF (_cmake_cxx_compiler_name STREQUAL "xlC")
   # IBM Visual Age:
   SET(REFLEX_COMPILER_CONFIG "config/compiler/VACpp")
ELSEIF (_cmake_cxx_compiler_name STREQUAL "CC")
   # Sun Workshop Compiler C++
   SET(MUREX_COMPILER_CONFIG "config/compiler/SunProCC")
ELSEIF (MSVC)
   # Microsoft Visual C++
   SET(REFLEX_COMPILER_CONFIG "config/compiler/VisualC")
ELSE (MSVC)
   # this must come last - generate a warning if we don't
   # recognise the compiler:
   MESSAGE(STATUS "Unknown compiler - please configure and report the results to Reflex")
ENDIF (CMAKE_COMPILER_IS_GNUCXX)
