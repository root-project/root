############################################################################
# CMakeLists.txt file for building ROOT graf2d/asimage package
# @author Pere Mato, CERN
############################################################################

if(cocoa)  # Not needed to locate PNG, JPG, TIFF libraries
  set(ASEXTRA_LIBRARIES)
endif()

#---Build ASImage/ASImageGui libraries---------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist

include_directories(${FREETYPE_INCLUDE_DIRS} ${AFTERIMAGE_INCLUDE_DIR} ${X11_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__ASImage TASImage.h TASImagePlugin.h TASPluginGS.h MODULE ASImage LINKDEF LinkDef.h  OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(ASImage TASImage.cxx TASPluginGS.cxx  G__ASImage.cxx
                    LIBRARIES Core ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES}
                    ${ZLIB_LIBRARY} DEPENDENCIES Graf)
if(builtin_afterimage)
  ROOT_ADD_BUILTIN_DEPENDENCIES(ASImage AFTERIMAGE)
endif()
if(builtin_freetype)
  ROOT_ADD_BUILTIN_DEPENDENCIES(ASImage FREETYPE)
endif()

ROOT_GENERATE_DICTIONARY(G__ASImageGui TASPaletteEditor.h MODULE ASImageGui LINKDEF LinkDefGui.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(ASImageGui TASPaletteEditor.cxx G__ASImageGui.cxx
                    LIBRARIES ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES}
                    ${ZLIB_LIBRARY} DEPENDENCIES Gui ASImage)

ROOT_INSTALL_HEADERS()
