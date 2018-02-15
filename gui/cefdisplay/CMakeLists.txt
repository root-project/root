############################################################################
# CMakeLists.txt file for building ROOT gui/cefdisplay package
############################################################################

set(libname ROOTCefDisplay)

ROOT_GLOB_SOURCES(CEF_sources src/simple_app.cxx src/base_handler.cxx src/osr_handler.cxx src/gui_handler.cxx)

if(ROOT_ARCHITECTURE MATCHES macosx)
   include_directories($ENV{CEF_PATH} inc)
   ROOT_GLOB_SOURCES(CEF_platform src/gui_handler_mac.mm)
   set(CEF_DLL_WRAPPER $ENV{CEF_PATH}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_MAIN "src/cef_main.cxx")
   set(CEF_RESOURCES $ENV{CEF_PATH}/Release/Chromium\ Embedded\ Framework.framework/Resources)
   configure_file(${CEF_RESOURCES}/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
   configure_file(${CEF_RESOURCES}/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
   configure_file(${CEF_RESOURCES}/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
   configure_file(${CEF_RESOURCES}/v8_context_snapshot.bin "../../bin/v8_context_snapshot.bin" COPYONLY)
   set(CEF_LIBRARY $ENV{CEF_PATH}/Release/Chromium\ Embedded\ Framework.framework/Chromium\ Embedded\ Framework)
   set(CEF_LIB_DEPENDENCY "-framework Foundation")
else ()
   include_directories($ENV{CEF_PATH} inc)
#   add_definitions(-DWEBGUI_WITH_CEF)
   ROOT_GLOB_SOURCES(CEF_platform src/gui_handler_linux.cxx)
   configure_file($ENV{CEF_PATH}/Resources/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
   configure_file($ENV{CEF_PATH}/Release/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
   set(CEF_LIBRARY $ENV{CEF_PATH}/Release/libcef.so)
   set(CEF_DLL_WRAPPER $ENV{CEF_PATH}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_LIB_DEPENDENCY ${X11_LIBRARIES})
   set(CEF_MAIN "src/cef_main.cxx")
endif()


ROOT_OBJECT_LIBRARY(ROOTCefDisplayObjs ${CEF_sources} ${CEF_platform})

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:ROOTCefDisplayObjs> 
                    LIBRARIES ${CMAKE_DL_LIBS} ${CEF_LIBRARY} ${CEF_DLL_WRAPPER} ${CEF_LIB_DEPENDENCY}
                    DEPENDENCIES RHTTP)
                    
# ROOT_INSTALL_HEADERS()

ROOT_EXECUTABLE(cef_main ${CEF_MAIN} LIBRARIES ${CEF_LIBRARY} ${CEF_DLL_WRAPPER})
