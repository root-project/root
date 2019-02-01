############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

set(Macosx_dict_headers
  TMacOSXSystem.h
)

ROOT_OBJECT_LIBRARY(Macosx
  src/CocoaUtils.mm
  src/TMacOSXSystem.mm
  G__Macosx.cxx
)

set_source_files_properties(
  src/CocoaUtils.mm
  src/TMacOSXSystem.mm
  PROPERTIES COMPILE_FLAGS -ObjC++
)

ROOT_GENERATE_DICTIONARY(G__Macosx
  ${Macosx_dict_headers}
  STAGE1
  MULTIDICT
  MODULE
    Core
  LINKDEF
    LinkDef.h
)

ROOT_INSTALL_HEADERS()
