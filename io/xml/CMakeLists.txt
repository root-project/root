# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/xml package
############################################################################

set(XMLIO_HEADERS
  TBufferXML.h
  TKeyXML.h
  TXMLEngine.h
  TXMLFile.h
  TXMLPlayer.h
  TXMLSetup.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(XMLIO
  HEADERS
    ${XMLIO_HEADERS}
  SOURCES
     src/TBufferXML.cxx
     src/TKeyXML.cxx
     src/TXMLEngine.cxx
     src/TXMLFile.cxx
     src/TXMLPlayer.cxx
     src/TXMLSetup.cxx
  DEPENDENCIES RIO)


set(RELATIVE_INC_HEADERS ${XMLIO_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(XMLIO
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
