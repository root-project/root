# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/vecgeom package
# @author Mihaela Gheata, CERN
############################################################################

include_directories(AFTER SYSTEM ${VECGEOM_INCLUDE_DIRS})

set(VECGEOM_HEADERS
    TGeoVGConverter.h
    TGeoVGShape.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(ConverterVG
  HEADERS
    ${VECGEOM_HEADERS}
  SOURCES
    src/TGeoVGConverter.cxx
    src/TGeoVGShape.cxx
  LIBRARIES
    ${VECGEOM_LIBRARIES}
    ${CMAKE_DL_LIBS}
  DEPENDENCIES
    Geom
)

set(RELATIVE_INC_HEADERS ${VECGEOM_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(ConverterVG
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
