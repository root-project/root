############################################################################
# CMakeLists.txt file for building ROOT net/ldap package
############################################################################

ROOT_USE_PACKAGE(net/net)
include_directories(${LDAP_INCLUDE_DIRS})
add_definitions(-DLDAP_DEPRECATED)

ROOT_GENERATE_DICTIONARY(G__LDAP *.h MODULE RLDAP LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RLDAP *.cxx G__LDAP.cxx LIBRARIES Core ${LDAP_LIBRARIES} DEPENDENCIES Net)
ROOT_INSTALL_HEADERS()
