############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

set(Macosx_dict_headers
  TMacOSXSystem.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(Macosx
  src/CocoaUtils.mm
  src/TMacOSXSystem.mm
)

target_compile_options(Macosx PRIVATE -ObjC++)

ROOT_INSTALL_HEADERS()
