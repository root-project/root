IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
   # linux:
   SET(REFLEX_PLATFORM_CONFIG "config/platform/Linux")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "AIX")
   # IBM AIX:
   SET(REFLEX_PLATFORM_CONFIG "config/platform/AIX")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "SunOS")
   # Solaris
   SET(REFLEX_PLATFORM_CONFIG "config/platform/Solaris")
ELSEIF (CYGWIN)
   # cygwin is not win32:
   SET(REFLEX_PLATFORM_CONFIG "config/platform/Cygwin")
ELSEIF (APPLE)
   # MacOS:
   SET(REFLEX_PLATFORM_CONFIG "config/platform/MacOS")
ELSEIF (WIN32)
   # win32:
   SET(REFLEX_PLATFORM_CONFIG "config/platform/Win32")
ELSE (WIN32)
   # this must come last - generate a warning if we don't
   # recognise the platform:
   MESSAGE(STATUS "Unknown platform - please configure and report the results to Reflex")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
