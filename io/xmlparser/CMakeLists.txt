############################################################################
# CMakeLists.txt file for building ROOT io/xmlparser package
# @author Pere Mato, CERN
############################################################################

include_directories(${LIBXML2_INCLUDE_DIR})
add_definitions(${LIBXML2_DEFINITIONS})

if(WIN32)
  set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARIES} ws2_32.lib)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(XMLParser
                              LIBRARIES ${LIBXML2_LIBRARIES}
                              DEPENDENCIES Core)
