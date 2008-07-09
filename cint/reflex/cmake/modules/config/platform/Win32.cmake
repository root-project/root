IF (MINGW)
   MESSAGE(FATAL_ERROR "MinGW is NOT supported, please use MSVC to build Reflex.")
ENDIF (MINGW)

# Set a default build type to debug if no build type already set.
IF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Debug)
ENDIF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
