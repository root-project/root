############################################################################
# CMakeLists.txt file for building ROOT core/winnt package
############################################################################

set(Winnt_dict_headers
  TWinNTSystem.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(Winnt
  src/TWin32SplashThread.cxx
  src/TWinNTSystem.cxx
  src/Win32Splash.cxx
)

ROOT_INSTALL_HEADERS()
