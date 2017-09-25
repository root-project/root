############################################################################
# CMakeLists.txt file for building ROOT net/alien package
# @author Pere Mato, CERN
############################################################################

include_directories(${ALIEN_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(RAliEn
                              LIBRARIES ${ALIEN_LIBRARIES}
                              DEPENDENCIES XMLIO Netx Tree Proof)
