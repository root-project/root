############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

set(libname ROOTWebDisplay)

ROOT_GLOB_HEADERS(WebDisplay_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)
ROOT_GLOB_SOURCES(WebDisplay_sources src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__${libname} ${WebDisplay_headers} MODULE ${libname} LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h OPTIONS DEPENDENCIES RHTTP)

set(CEF_LIBRARY "")
set(CEF_LIB_DEPENDENCY "")
set(CEF_DLL_WRAPPER "")
set(CEF_SRC "")
set(CEF_RESOURCES "")

if(DEFINED ENV{CEF_PATH})
if(EXISTS $ENV{CEF_PATH})
if(ROOT_ARCHITECTURE MATCHES macosx)
   include_directories($ENV{CEF_PATH})
   ROOT_GLOB_SOURCES(CEF_sources cef/simple_app.cxx cef/base_handler.cxx cef/osr_handler.cxx cef/gui_handler.cxx cef/gui_handler_mac.mm)
   set(CEF_DLL_WRAPPER $ENV{CEF_PATH}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_MAIN "cef/cef_main.cxx")
   set(CEF_RESOURCES $ENV{CEF_PATH}/Release/Chromium\ Embedded\ Framework.framework/Resources)
   configure_file(${CEF_RESOURCES}/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
   configure_file(${CEF_RESOURCES}/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
   configure_file(${CEF_RESOURCES}/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
   configure_file(${CEF_RESOURCES}/v8_context_snapshot.bin "../../bin/v8_context_snapshot.bin" COPYONLY)
   set(CEF_LIBRARY $ENV{CEF_PATH}/Release/Chromium\ Embedded\ Framework.framework/Chromium\ Embedded\ Framework)
   set(CEF_LIB_DEPENDENCY "-framework Foundation")
else ()
   include_directories($ENV{CEF_PATH})
#   add_definitions(-DWEBGUI_WITH_CEF)
   ROOT_GLOB_SOURCES(CEF_sources cef/simple_app.cxx cef/base_handler.cxx cef/osr_handler.cxx cef/gui_handler.cxx cef/gui_handler_linux.cxx)
   configure_file($ENV{CEF_PATH}/Resources/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
   set(CEF_LIBRARY $ENV{CEF_PATH}/Release/libcef.so)
   set(CEF_DLL_WRAPPER $ENV{CEF_PATH}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_LIB_DEPENDENCY ${X11_LIBRARIES})
   set(CEF_MAIN "cef/cef_main.cxx")
endif()
endif()
endif()

# ROOT_LINKER_LIBRARY(${libname}
#                    HEADERS ${WebDisplay_headers}
#                    SOURCES ${WebDisplay_sources} ${CEF_SRC}
#                    LIBRARIES Core ${CEF_LIBRARY} ${CEF_DLL_WRAPPER} ${CEF_LIB_DEPENDENCY}
#                    DEPENDENCIES RHTTP)

# ROOT_LINKER_LIBRARY(${libname} *.cxx src/ DEPENDENCIES RHTTP)

ROOT_OBJECT_LIBRARY(ROOTWebDisplayObjs G__ROOTWebDisplay.cxx ${WebDisplay_sources} ${CEF_sources})

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:ROOTWebDisplayObjs> $<TARGET_OBJECTS:RootPcmObjs>
                    LIBRARIES ${CMAKE_DL_LIBS} ${CEF_LIBRARY} ${CEF_DLL_WRAPPER} ${CEF_LIB_DEPENDENCY}
                    DEPENDENCIES RHTTP)
ROOT_INSTALL_HEADERS()

if(DEFINED CEF_MAIN)
  ROOT_EXECUTABLE(cef_main ${CEF_MAIN} LIBRARIES ${CEF_LIBRARY} ${CEF_DLL_WRAPPER})
endif()
