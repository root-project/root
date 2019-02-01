############################################################################
# CMakeLists.txt file for building ROOT core/winnt package
############################################################################

set(Winnt_dict_headers
  TWinNTSystem.h
)

ROOT_OBJECT_LIBRARY(Winnt
  src/TWin32SplashThread.cxx
  src/TWinNTSystem.cxx
  src/Win32Splash.cxx
  G__Windows.cxx
)

ROOT_GENERATE_DICTIONARY(G__Windows
  ${Winnt_dict_headers}
  STAGE1
  MULTIDICT
  MODULE
    Core
  LINKDEF
    LinkDef.h
)

ROOT_INSTALL_HEADERS()
