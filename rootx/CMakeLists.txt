############################################################################
# CMakeLists.txt file for building ROOT rootx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
if(x11)
  include_directories(${X11_INCLUDE_DIR})
  ROOT_EXECUTABLE(root *.cxx  ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c
                  LIBRARIES ${X11_LIBRARIES} ${X11_Xpm_LIB} ${X11_Xft_LIB} ${X11_Xext_LIB})
elseif(cocoa)
  ROOT_EXECUTABLE(root rootx.cxx rootxx-cocoa.mm LIBRARIES "-framework Cocoa")
endif()

