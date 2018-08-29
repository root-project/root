############################################################################
# CMakeLists.txt file for building ROOT core/winnt package
############################################################################

set(sources src/TWin32SplashThread.cxx
   src/TWinNTSystem.cxx
   src/Win32Splash.cxx)

set(Winnt_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/TWinNTSystem.h PARENT_SCOPE)

include_directories(${CMAKE_SOURCE_DIR}/net/net/inc)

ROOT_OBJECT_LIBRARY(Winnt ${sources})

ROOT_INSTALL_HEADERS()
