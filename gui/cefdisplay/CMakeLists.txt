############################################################################
# CMakeLists.txt file for building ROOT gui/cefdisplay package
############################################################################

set(libname ROOTCefDisplay)

set(CEF_sources src/simple_app.cxx src/base_handler.cxx src/osr_handler.cxx src/gui_handler.cxx)

set(CEF_root $ENV{CEF_PATH})

if(ROOT_ARCHITECTURE MATCHES macosx)
   set(CEF_platform src/gui_handler_mac.mm)
   set(CEF_RESOURCES $ENV{CEF_PATH}/Release/Chromium\ Embedded\ Framework.framework/Resources)
   set(CEF_BLOBS ${CEF_RESOURCES})
   set(CEF_LIBRARY ${CEF_root}/Release/Chromium\ Embedded\ Framework.framework/Chromium\ Embedded\ Framework)
   set(CEF_LIB_DEPENDENCY "-framework Foundation")
   set(CEF_DLL_WRAPPER ${CEF_root}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
elseif(MSVC)
   set(CEF_platform src/gui_handler_win.cc)
   set(CEF_RESOURCES ${CEF_root}/Resources)
   set(CEF_BLOBS ${CEF_root}/Release)
   set(CEF_LIBRARY ${CEF_root}/Release/libcef.lib)
   set(CEF_DLL_WRAPPER ${CEF_root}/build/libcef_dll_wrapper/Release/libcef_dll_wrapper.lib)
   configure_file(${CEF_root}/Release/libcef.dll "../../bin/libcef.dll" COPYONLY)
else ()
   set(CEF_platform src/gui_handler_linux.cxx)
   set(CEF_RESOURCES ${CEF_root}/Resources)
   set(CEF_BLOBS ${CEF_root}/Release)
   set(CEF_LIBRARY ${CEF_root}/Release/libcef.so)
   set(CEF_DLL_WRAPPER ${CEF_root}/build/libcef_dll_wrapper/libcef_dll_wrapper.a)
   set(CEF_LIB_DEPENDENCY ${X11_LIBRARIES})
endif()

include_directories($ENV{CEF_PATH} inc)
set(CEF_MAIN src/cef_main.cxx)

configure_file(${CEF_RESOURCES}/icudtl.dat "../../bin/icudtl.dat" COPYONLY)
configure_file(${CEF_BLOBS}/natives_blob.bin "../../bin/natives_blob.bin" COPYONLY)
configure_file(${CEF_BLOBS}/snapshot_blob.bin "../../bin/snapshot_blob.bin" COPYONLY)
configure_file(${CEF_BLOBS}/v8_context_snapshot.bin "../../bin/v8_context_snapshot.bin" COPYONLY)

ROOT_LINKER_LIBRARY(${libname} ${CEF_sources} ${CEF_platform} 
                    LIBRARIES ${CMAKE_DL_LIBS} ${CEF_LIBRARY} ${CEF_DLL_WRAPPER} ${CEF_LIB_DEPENDENCY}
                    DEPENDENCIES RHTTP)

ROOT_EXECUTABLE(cef_main ${CEF_MAIN} LIBRARIES ${CEF_LIBRARY} ${CEF_DLL_WRAPPER})
