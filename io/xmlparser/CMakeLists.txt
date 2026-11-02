# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/xmlparser package
# @author Pere Mato, CERN
############################################################################

set(XMLPARSER_HEADERS
  TDOMParser.h
  TSAXParser.h
  TXMLAttr.h
  TXMLDocument.h
  TXMLNode.h
  TXMLParser.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(XMLParser
  HEADERS
     ${XMLPARSER_HEADERS}
  SOURCES
     src/TDOMParser.cxx
     src/TSAXParser.cxx
     src/TXMLAttr.cxx
     src/TXMLDocument.cxx
     src/TXMLNode.cxx
     src/TXMLParser.cxx
  DEPENDENCIES
    Core
)

set(RELATIVE_INC_HEADERS ${XMLPARSER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(XMLParser
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

# not needed to propagate includes to rootcling
target_include_directories(XMLParser PRIVATE
   ${LIBXML2_INCLUDE_DIR}
   ${LIBXML2_INCLUDE_DIRS}
)

target_link_libraries(XMLParser
  PRIVATE
    ${LIBXML2_LIBRARIES}
    $<$<PLATFORM_ID:Windows>:ws2_32.lib>
)
