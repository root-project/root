############################################################################
# CMakeLists.txt file for building ROOT net/alien package
# @author Pere Mato, CERN
############################################################################

include_directories(${ALIEN_INCLUDE_DIR} ${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})
ROOT_STANDARD_LIBRARY_PACKAGE(RAliEn
                              LIBRARIES ${ALIEN_LIBRARIES} ${XROOTD_LIBRARIES}
                              DEPENDENCIES XMLIO Netx NetxNG Tree Proof)
