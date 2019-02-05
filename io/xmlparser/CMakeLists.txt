############################################################################
# CMakeLists.txt file for building ROOT io/xmlparser package
# @author Pere Mato, CERN
############################################################################

include_directories(${LIBXML2_INCLUDE_DIR} ${LIBXML2_INCLUDE_DIRS})
add_definitions(${LIBXML2_DEFINITIONS})

if(WIN32)
  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES} ws2_32.lib)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(XMLParser
                              HEADERS
                                 TDOMParser.h
                                 TSAXParser.h
                                 TXMLAttr.h
                                 TXMLDocument.h
                                 TXMLNode.h
                                 TXMLParser.h
                              SOURCES
                                 src/TDOMParser.cxx
                                 src/TSAXParser.cxx
                                 src/TXMLAttr.cxx
                                 src/TXMLDocument.cxx
                                 src/TXMLNode.cxx
                                 src/TXMLParser.cxx
                              LIBRARIES ${LIBXML2_LIBRARIES}
                              DEPENDENCIES Core)
