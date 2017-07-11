############################################################################
# CMakeLists.txt file for building ROOT net/ldap package
############################################################################

include_directories(${LDAP_INCLUDE_DIRS})
add_definitions(-DLDAP_DEPRECATED)

if(MACOSX_LDAP_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_GENERATE_DICTIONARY(G__RLDAP *.h MODULE RLDAP LINKDEF LinkDef.h DEPENDENCIES Net)

ROOT_LINKER_LIBRARY(RLDAP *.cxx G__RLDAP.cxx LIBRARIES Core ${LDAP_LIBRARIES} DEPENDENCIES Net)
ROOT_INSTALL_HEADERS()
