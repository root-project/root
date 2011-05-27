############################################################################
# CMakeLists.txt file for building ROOT io/xmlparser package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(io/io)
include_directories(${LIBXML2_INCLUDE_DIR})
add_definitions(${LIBXML2_DEFINITIONS})

ROOT_GENERATE_DICTIONARY(G__XMLParser *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(XMLParser LINKDEF LinkDef.h )

ROOT_LINKER_LIBRARY(XMLParser *.cxx G__XMLParser.cxx  LIBRARIES ${LIBXML2_LIBRARIES} DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()

