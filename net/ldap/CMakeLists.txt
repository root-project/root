############################################################################
# CMakeLists.txt file for building ROOT net/ldap package
############################################################################

include_directories(${LDAP_INCLUDE_DIRS})
add_definitions(-DLDAP_DEPRECATED)

if(MACOSX_LDAP_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RLDAP
  HEADERS
    CintLDAP.h
    TLDAPAttribute.h
    TLDAPEntry.h
    TLDAPResult.h
    TLDAPServer.h
  SOURCES
    src/TLDAPAttribute.cxx
    src/TLDAPEntry.cxx
    src/TLDAPResult.cxx
    src/TLDAPServer.cxx
  LIBRARIES
    ${LDAP_LIBRARIES}
  DEPENDENCIES
    Net
    Core
  )
