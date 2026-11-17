# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/asimage package
# @author Pere Mato, CERN
# @author Sergey Linev, GSI
############################################################################

if(asimage)

  set(ASIMAGE_HEADERS 
    TASImage.h
    TASImagePlugin.h
    TASPluginGS.h
    TASPngWriter.h
  )
  ROOT_STANDARD_LIBRARY_PACKAGE(ASImage
    NO_GLOB_HEADERS
    HEADERS
      ${ASIMAGE_HEADERS}
    SOURCES
      TASImage.cxx
      TASPluginGS.cxx
      TASPngWriter.cxx
    DICTIONARY_OPTIONS
      -writeEmptyRootPCM
    LIBRARIES
      libAfterImage
    DEPENDENCIES
      Core
      Graf
      Postscript
  )
  
  set(RELATIVE_INC_HEADERS ${ASIMAGE_HEADERS})
  list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
  target_sources(ASImage
  PRIVATE
    FILE_SET private_header_files
    TYPE HEADERS
    BASE_DIRS inc/
    FILES
      ${RELATIVE_INC_HEADERS}
      inc/LinkDef.h
  )

  if (x11)
    target_link_libraries(ASImage PRIVATE X11::X11)
  endif()

endif()

set(ASIMAGEUTILS_HEADERS TASUtils.h)

ROOT_STANDARD_LIBRARY_PACKAGE(ASImageUtils
  NO_GLOB_HEADERS
  HEADERS
    ${ASIMAGEUTILS_HEADERS}
  SOURCES
    TASUtils.cxx
  LINKDEF
    LinkDefUtils.h
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
)

set(RELATIVE_INC_HEADERS ${ASIMAGEUTILS_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ASImageUtils
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDefUtils.h
)

if(asimage)

  target_link_libraries(ASImageUtils PUBLIC ASImage)

  set(ASIMAGEGUI_HEADERS TASPaletteEditor.h)
  ROOT_STANDARD_LIBRARY_PACKAGE(ASImageGui
    NO_GLOB_HEADERS
    HEADERS
      ${ASIMAGEGUI_HEADERS}
    SOURCES
      TASPaletteEditor.cxx
    LINKDEF
      LinkDefGui.h
    DICTIONARY_OPTIONS
      -writeEmptyRootPCM
    LIBRARIES
      libAfterImage
    DEPENDENCIES
      Gui
      ASImage
  )
  set(RELATIVE_INC_HEADERS ${ASIMAGEGUI_HEADERS})
  list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
  target_sources(ASImageGui
  PRIVATE
    FILE_SET private_header_files
    TYPE HEADERS
    BASE_DIRS inc/
    FILES
      ${RELATIVE_INC_HEADERS}
      inc/LinkDefGui.h
  )

  ROOT_ADD_TEST_SUBDIRECTORY(test)

else()
  target_compile_definitions(ASImageUtils PRIVATE NO_ASIMAGE)
endif()
