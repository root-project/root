############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm)
ROOT_GLOB_SOURCES(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)

set(Macosx_dict_headers ${headers} PARENT_SCOPE)

if(cxxmodules)
  # FIXME: Disable modules for ObjC/ObjC++.
  # ROOT top-most module, cannot be built implicitly in ObjC/ObjC++ TU context
  # because of issues in TBonjourBrowser (when incluing dns_sd.h) on OSX.
  string(REPLACE "${ROOT_CXXMODULES_CXXFLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "${ROOT_CXXMODULES_CFLAGS}" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif(cxxmodules)

ROOT_OBJECT_LIBRARY(Macosx ${sources})
set_source_files_properties(${sources} COMPILE_FLAGS "-ObjC++ -std=c++11")

ROOT_INSTALL_HEADERS()
