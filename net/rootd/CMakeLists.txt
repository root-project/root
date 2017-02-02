############################################################################
# CMakeLists.txt file for building ROOT net/rootd package
# @author Pere Mato, CERN
############################################################################

ROOT_EXECUTABLE(rootd *.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c
                      LIBRARIES rpdutil rsa ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} ${CRYPTLIBS}
                      BUILTINS OPENSSL)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../rpdutils/res)
