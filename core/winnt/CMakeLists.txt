# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

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

target_include_directories(Winnt PRIVATE ../foundation/res)

ROOT_INSTALL_HEADERS()
