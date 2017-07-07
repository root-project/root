############################################################################
# CMakeLists.txt file for building ROOT io/xmlparser package
# @author Pere Mato, CERN
############################################################################

include_directories(${LIBXML2_INCLUDE_DIR})
add_definitions(${LIBXML2_DEFINITIONS})

ROOT_GENERATE_DICTIONARY(G__XMLParser *.h MODULE XMLParser LINKDEF LinkDef.h DEPENDENCIES Core)

if(WIN32)
  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES} ws2_32.lib)
endif()

ROOT_LINKER_LIBRARY(XMLParser *.cxx G__XMLParser.cxx  LIBRARIES ${LIBXML2_LIBRARIES} DEPENDENCIES Core)
ROOT_INSTALL_HEADERS()

