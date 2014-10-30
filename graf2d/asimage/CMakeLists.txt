############################################################################
# CMakeLists.txt file for building ROOT graf2d/asimage package
# @author Pere Mato, CERN
############################################################################

#---Build afterimage library------------------------------------------------
if(builtin_afterimage)
  include(BuildAfterImage.cmake)
  set(AFTERIMAGE_INCLUDE_DIR ${CMAKE_BINARY_DIR}/graf2d/asimage/libAfterImage)
  if(WIN32)
    set(AFTERIMAGE_LIBRARIES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAfterImage.lib)
  else()
    set(AFTERIMAGE_LIBRARIES ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libAfterImage.a)
  endif()
endif()

if(cocoa)  # Not needed to locate PNG, JPG, TIFF libraries
  set(ASEXTRA_LIBRARIES)
endif()

#---Build ASImage/ASImageGui libraries---------------------------------------
ROOT_USE_PACKAGE(graf2d/graf)
ROOT_USE_PACKAGE(graf2d/x11)
ROOT_USE_PACKAGE(gui/gui)
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist

include_directories(${FREETYPE_INCLUDE_DIRS} ${AFTERIMAGE_INCLUDE_DIR} ${X11_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__ASImage TASImage.h TASImagePlugin.h TASPluginGS.h MODULE ASImage LINKDEF LinkDef.h  OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(ASImage TASImage.cxx TASPluginGS.cxx  G__ASImage.cxx
                    LIBRARIES Core ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES}
                    ${ZLIB_LIBRARY} DEPENDENCIES Graf)
if(builtin_afterimage)
  add_dependencies(ASImage AFTERIMAGE)
endif()
if(builtin_freetype)
  add_dependencies(ASImage FREETYPE)
endif()

ROOT_GENERATE_DICTIONARY(G__ASImageGui TASPaletteEditor.h MODULE ASImageGui LINKDEF LinkDefGui.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(ASImageGui TASPaletteEditor.cxx G__ASImageGui.cxx
                    LIBRARIES ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES}
                    ${ZLIB_LIBRARY} DEPENDENCIES Gui ASImage)

ROOT_INSTALL_HEADERS()
