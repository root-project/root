# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/win32gdk package
# @author Pere Mato, CERN
############################################################################

set(iconvlib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/iconv-1.3.lib)
set(iconvdll  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/iconv-1.3.dll)
set(iconvliba ${CMAKE_BINARY_DIR}/builtins/win32gdk/iconv/iconv-1.3.lib)
set(iconvdlla ${CMAKE_BINARY_DIR}/builtins/win32gdk/iconv/iconv-1.3.dll)

set(gliblib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/glib-1.3.lib)
set(glibdll  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/glib-1.3.dll)
set(glibliba ${CMAKE_BINARY_DIR}/builtins/win32gdk/glib/glib-1.3.lib)
set(glibdlla ${CMAKE_BINARY_DIR}/builtins/win32gdk/glib/glib-1.3.dll)

set(gdklib   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/gdk-1.3.lib)
set(gdkdll   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gdk-1.3.dll)
set(gdkliba  ${CMAKE_BINARY_DIR}/builtins/win32gdk/gdk/gdk-1.3.lib)
set(gdkdlla  ${CMAKE_BINARY_DIR}/builtins/win32gdk/gdk/gdk-1.3.dll)

file(GLOB iconvfiles "${CMAKE_SOURCE_DIR}/builtins/win32gdk/iconv/*.*")
file(GLOB_RECURSE glibfiles "${CMAKE_SOURCE_DIR}/builtins/win32gdk/glib/*.*")
file(GLOB_RECURSE gdkfiles "${CMAKE_SOURCE_DIR}/builtins/win32gdk/gdk/*.*")

if(CMAKE_GENERATOR MATCHES Ninja)
  set(nmcxxflags ${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}})
  if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(nmcxxflags "${nmcxxflags}" DEBUG=1)
  endif()
else()
  if(winrtdebug)
    set(nmcxxflags "${CMAKE_CXX_FLAGS_DEBUG}" DEBUG=1)
  else()
    set(nmcxxflags "${CMAKE_CXX_FLAGS_RELEASE}")
  endif()
endif()
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(nmcxxflags "${nmcxxflags} -D_WIN64")
endif()

# Copy to binary dir to avoid polluting source directory is done as part of the iconv step directly
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/builtins/)

# Build iconv, was forked from https://github.com/winlibs/libiconv/blob/libiconv-1.14/MSVC9/libiconv.rc
# vs https://github.com/root-project/root/commit/11e6c04d7ee196c794d19ae43ad4d1ad57786d3d#diff-9cbb57072c182c295932b31f4421982d0ac9d1ca2d545bd065509aba4ecfdfce
# and only slightly modified by ROOT
					
add_custom_command( OUTPUT ${iconvliba} ${iconvdlla}
					COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/builtins/win32gdk ${CMAKE_BINARY_DIR}/builtins/win32gdk
                    COMMAND echo "*** Building ${iconvliba}"
					COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/builtins/win32gdk/iconv
							nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
					DEPENDS ${iconvfiles} ${glibfiles} ${gdkfiles} )

add_custom_command( OUTPUT ${iconvlib} ${iconvdll}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${iconvliba} ${iconvlib}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${iconvdlla} ${iconvdll}
					DEPENDS ${iconvliba} ${iconvdlla} )

add_custom_target(iconv DEPENDS ${iconvlib})
set_target_properties(iconv PROPERTIES FOLDER Builtins)

# Build glib, was forked from https://gitlab.gnome.org/GNOME/glib/-/tags/GLIB_1_3_9
# vs https://github.com/root-project/root/commit/11e6c04d7ee196c794d19ae43ad4d1ad57786d3d#diff-9cbb57072c182c295932b31f4421982d0ac9d1ca2d545bd065509aba4ecfdfce
# and only slightly modified by ROOT

add_custom_command( OUTPUT ${glibliba} ${glibdlla}
					COMMAND echo "*** Building ${glibliba}"
					COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/builtins/win32gdk/glib
							nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
					DEPENDS iconv ${glibfiles} )

add_custom_command( OUTPUT ${gliblib} ${glibdll}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${glibliba} ${gliblib}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${glibdlla} ${glibdll}
					DEPENDS ${glibliba} ${glibdlla})

add_custom_target(glib DEPENDS ${gliblib})
set_target_properties(glib PROPERTIES FOLDER Builtins)

# Build gdk, was forked from https://github.com/GNOME/gtk/blob/GTK_1_3_1/gdk/win32/rc/gdk.rc
# vs https://github.com/root-project/root/commit/11e6c04d7ee196c794d19ae43ad4d1ad57786d3d#diff-9cbb57072c182c295932b31f4421982d0ac9d1ca2d545bd065509aba4ecfdfce
# and only slightly modified by ROOT

add_custom_command( OUTPUT ${gdkliba}
					COMMAND echo "*** Building ${gdkliba}"
					COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/builtins/win32gdk/gdk/win32
							nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
					COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/builtins/win32gdk/gdk/
							nmake -nologo -f makefile.msc NMCXXFLAGS=${nmcxxflags} VC_MAJOR=${VC_MAJOR}
					DEPENDS glib ${gdkfiles} )

add_custom_command( OUTPUT ${gdklib} ${gdkdll}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${gdkliba} ${gdklib}
					COMMAND ${CMAKE_COMMAND} -E copy_if_different ${gdkdlla} ${gdkdll}
					DEPENDS ${gdkliba} )

add_custom_target(GDKLIB DEPENDS ${gdklib})
set_target_properties(GDKLIB PROPERTIES FOLDER Builtins)

ROOT_STANDARD_LIBRARY_PACKAGE(Win32gdk
  HEADERS
    TGWin32.h
    TGWin32GL.h
  SOURCES
    ../gifencode/gifdecode.cxx
    ../gifencode/gifencode.cxx
    src/TGWin32.cxx
    src/TGWin32GL.cxx
    src/TGWin32ProxyBase.cxx
    src/TGWin32VirtualGLProxy.cxx
    src/TGWin32VirtualXProxy.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    Freetype::Freetype
    ${gdklib}
    ${gliblib}
    Glu32.lib
    Opengl32.lib
  DEPENDENCIES
    Core
    Graf
)

target_include_directories(Win32gdk PRIVATE ../gifencode)
target_include_directories(Win32gdk PRIVATE ${CMAKE_SOURCE_DIR}/builtins/win32gdk) # this should be instead dealt via PUBLIC target_include_directories of gdklib and gliblib targets

add_dependencies(Win32gdk GDKLIB glib)
install(FILES ${gdkdlla} ${glibdlla} ${iconvdlla} DESTINATION ${CMAKE_INSTALL_BINDIR})
