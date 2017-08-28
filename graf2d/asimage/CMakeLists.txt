############################################################################
# CMakeLists.txt file for building ROOT graf2d/asimage package
# @author Pere Mato, CERN
############################################################################

if(cocoa)  # Not needed to locate PNG, JPG, TIFF libraries
  set(ASEXTRA_LIBRARIES)
endif()

#---Build ASImage/ASImageGui libraries---------------------------------------
include_directories(${FREETYPE_INCLUDE_DIRS} ${AFTERIMAGE_INCLUDE_DIR} ${X11_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(ASImage
                              NO_INSTALL_HEADERS
                              HEADERS TASImage.h TASImagePlugin.h TASPluginGS.h
                              SOURCES TASImage.cxx TASPluginGS.cxx
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES Core ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES} ${ZLIB_LIBRARIES}
                              DEPENDENCIES Graf BUILTINS AFTERIMAGE)


ROOT_STANDARD_LIBRARY_PACKAGE(ASImageGui
                              NO_INSTALL_HEADERS
                              HEADERS TASPaletteEditor.h
                              SOURCES TASPaletteEditor.cxx
                              LINKDEF LinkDefGui.h
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Gui ASImage
                              LIBRARIES ${AFTERIMAGE_LIBRARIES} ${FREETYPE_LIBRARIES} ${ASEXTRA_LIBRARIES} ${X11_LIBRARIES} ${ZLIB_LIBRARIES})
ROOT_INSTALL_HEADERS()
