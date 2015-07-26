############################################################################
# CMakeLists.txt file for building ROOT net/globusauth package
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(net/auth)

include_directories(${GLOBUS_INCLUDE_DIRS})
if(ssl)
  include_directories(${OPENSSL_INCLUDE_DIR})
endif()

ROOT_LINKER_LIBRARY(GlobusAuth *.cxx  LIBRARIES Core ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} DEPENDENCIES RootAuth Net)
ROOT_INSTALL_HEADERS()
