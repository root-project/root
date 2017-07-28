############################################################################
# CMakeLists.txt file for building ROOT net/bonjour package
# @author Pere Mato, CERN
############################################################################

include_directories(${BONJOUR_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(Bonjour
                              LIBRARIES ${BONJOUR_LIBRARIES}
                              DEPENDENCIES Core)

