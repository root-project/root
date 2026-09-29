# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/winnt package
############################################################################

if(NOT WIN32)
  return()
endif()

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS TWinNTSystem.h)

target_sources(Core PRIVATE
  src/TWin32SplashThread.cxx
  src/TWinNTSystem.cxx
  src/Win32Splash.cxx
)

target_link_libraries(Core PRIVATE
  shell32.lib
  WSock32.lib
  Oleaut32.lib
  Iphlpapi.lib
  Bcrypt.lib
)

target_include_directories(Core PRIVATE inc)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Core
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          inc/TWin32SplashThread.h
          inc/TWinNTSystem.h
          inc/Win32Constants.h
          inc/Windows4Root.h
          inc/LinkDef.h
  )
endif()

ROOT_INSTALL_HEADERS()
