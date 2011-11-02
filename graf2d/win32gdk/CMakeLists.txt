############################################################################
# CMakeLists.txt file for building ROOT graf2d/win32gdk package
# @author Pere Mato, CERN
############################################################################
ROOT_USE_PACKAGE(gui/gui)
include_directories(${FREETYPE_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/gdk/src 
                    ${CMAKE_CURRENT_SOURCE_DIR}/gdk/src/gdk ${CMAKE_CURRENT_SOURCE_DIR}/gdk/src/glib)

set(gdklib   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/gdk-1.3.lib)
set(glib     ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/glib-1.3.lib)
set(gdkdll   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/gdk-1.3.dll)
set(gdkliba  ${CMAKE_CURRENT_BINARY_DIR}/gdk/src/gdk/gdk-1.3.lib)
set(gdkdlla  ${CMAKE_CURRENT_BINARY_DIR}/gdk/src/gdk/gdk-1.3.dll)
if(winrtdebug)
  set(nmcxxflags "${CMAKE_CXX_FLAGS}" DEBUG=1)
else()
  set(nmcxxflags "${CMAKE_CXX_FLAGS}")
endif()

add_custom_command( OUTPUT ${gdkliba}
                    COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/gdk gdk
					COMMAND cmake -E copy_if_different gdk/lib/glib-1.3.lib ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
					COMMAND cmake -E copy_if_different gdk/dll/glib-1.3.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}
					COMMAND cmake -E copy_if_different gdk/dll/iconv-1.3.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}
					COMMAND echo "*** Building ${gdkliba}"
		            COMMAND cmake -E chdir gdk/src/gdk/win32 
					        nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
		            COMMAND cmake -E chdir gdk/src/gdk 
					        nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
				  )

add_custom_command( OUTPUT ${gdklib} ${gdkdll}
                    COMMAND cmake -E copy_if_different ${gdkliba} ${gdklib}
					COMMAND cmake -E copy_if_different ${gdkdlla} ${gdkdll}
					DEPENDS ${gdkliba} )

add_custom_target(GDKLIB DEPENDS ${gdklib})
set_target_properties(GDKLIB PROPERTIES FOLDER Builtins)

ROOT_GENERATE_DICTIONARY(G__Win32gdk TGWin32.h TGWin32GL.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Win32gdk LINKDEF LinkDef.h DEPENDENCIES)

ROOT_LINKER_LIBRARY(Win32gdk *.cxx *.c G__Win32gdk.cxx LIBRARIES Cint Core Graf ${FREETYPE_LIBRARIES} 
                    ${gdklib} ${glib} Glu32.lib Opengl32.lib)
add_dependencies(Win32gdk GDKLIB)
if(builtin_freetype)
  add_dependencies(Win32gdk FREETYPE)
endif()
					
install(FILES  ${gdkdlla}
               gdk/dll/glib-1.3.dll
               gdk/dll/iconv-1.3.dll
               DESTINATION bin)

ROOT_INSTALL_HEADERS()
