############################################################################
# CMakeLists.txt file for building ROOT net/ldap package
############################################################################

include_directories(${LDAP_INCLUDE_DIRS})
add_definitions(-DLDAP_DEPRECATED)

if(MACOSX_LDAP_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RLDAP
                              LIBRARIES Core ${LDAP_LIBRARIES}
                              DEPENDENCIES Core Net)
