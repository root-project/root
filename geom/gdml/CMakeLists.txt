# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT geom/gdml package
############################################################################

set(GDML_HEADERS TGDMLParse.h TGDMLWrite.h)

set(RELATIVE_INC_HEADERS ${GDML_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(Gdml
  HEADERS
    ${GDML_HEADERS}
  SOURCES
    src/TGDMLParse.cxx
    src/TGDMLWrite.cxx
  DEPENDENCIES
    Geom
    XMLIO
    Hist
    RIO
)

target_sources(Gdml
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
