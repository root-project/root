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
    set(AFTERIMAGE_LIBRARIES "-L${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -lAfterImage")
  endif()
endif()

#---Build ASImage/ASImageGui libraries---------------------------------------
ROOT_USE_PACKAGE(graf2d/graf)
ROOT_USE_PACKAGE(graf2d/x11)
ROOT_USE_PACKAGE(gui/gui)
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist 

include_directories(${FREETYPE_INCLUDE_DIRS} ${AFTERIMAGE_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__ASImage TASImage.h TASImagePlugin.h TASPluginGS.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(ASImage LINKDEF LinkDef.h DEPENDENCIES Graf MathCore)
ROOT_LINKER_LIBRARY(ASImage TASImage.cxx TASPluginGS.cxx  G__ASImage.cxx LIBRARIES Cint Core
                    ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES}
					DEPENDENCIES Graf)
if(builtin_afterimage)
  add_dependencies(ASImage AFTERIMAGE)
endif()
if(builtin_freetype)
  add_dependencies(ASImage FREETYPE)
endif()

ROOT_GENERATE_DICTIONARY(G__ASImageGui TASPaletteEditor.h LINKDEF LinkDefGui.h)
ROOT_GENERATE_ROOTMAP(ASImageGui LINKDEF LinkDefGui.h DEPENDENCIES Graf Hist Gui ASImage RIO )
ROOT_LINKER_LIBRARY(ASImageGui TASPaletteEditor.cxx G__ASImageGui.cxx LIBRARIES  ASImage DEPENDENCIES Gui)

ROOT_INSTALL_HEADERS()
