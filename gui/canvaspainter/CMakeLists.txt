############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

set(libname ROOTCanvasPainter)

set(CEF_LIBRARY "")
set(CEF_LIB_DEPENDENCY "")
set(CEF_DLL_WRAPPER "")
set(CEF_SRC "")
ROOT_GLOB_SOURCES(sources v7/src/*.cxx)

if(DEFINED ENV{CEF_PATH})
if(EXISTS $ENV{CEF_PATH})
   include_directories($ENV{CEF_PATH})
#   add_definitions(-DWEBGUI_WITH_CEF)
   ROOT_GLOB_SOURCES(sources v7/src/*.cxx v7/cef/simple_app.cxx v7/cef/base_handler.cxx v7/cef/osr_handler.cxx v7/cef/gui_handler.cxx v7/cef/gui_handler_linux.cxx)
   configure_file($ENV{CEF_PATH}/Resources/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
   set(CEF_LIBRARY $ENV{CEF_PATH}/Release/libcef.so)
   set(CEF_DLL_WRAPPER $ENV{CEF_PATH}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_LIB_DEPENDENCY ${X11_LIBRARIES})
   set(CEF_MAIN "v7/cef/cef_main.cxx")
endif()
endif()

ROOT_LINKER_LIBRARY(${libname} ${sources} LIBRARIES Core ${CEF_LIBRARY} ${CEF_DLL_WRAPPER} ${CEF_LIB_DEPENDENCY} DEPENDENCIES Gpad RHTTP)

# ROOT_LINKER_LIBRARY(${libname} *.cxx v7/src/ DEPENDENCIES Gpad RHTTP)

if(DEFINED CEF_MAIN)
  ROOT_EXECUTABLE(cef_main ${CEF_MAIN} LIBRARIES ${CEF_LIBRARY} ${CEF_DLL_WRAPPER})
endif()
