# - Try to find the LDAP client libraries
# Once done this will define
#
#  LDAP_FOUND - system has libldap
#  LDAP_INCLUDE_DIR - the ldap include directory
#  LDAP_LIBRARY  libldap library
#  LBER_LIBRARY  liblber library
#  LDAP_LIBRARIES - libldap + liblber (if found) library


find_path(LDAP_INCLUDE_DIR NAMES ldap.h HINTS ${LDAP_DIR}/include $ENV{LDAP_DIR}/include)
find_library(LDAP_LIBRARY NAMES ldap HINTS ${LDAP_DIR}/lib $ENV{LDAP_DIR}/lib)
find_library(LBER_LIBRARY NAMES lber HINTS ${LDAP_DIR}/lib $ENV{LDAP_DIR}/lib)

set(LDAP_INCLUDE_DIRS ${LDAP_INCLUDE_DIR})
set(LDAP_LIBRARIES ${LDAP_LIBRARY} ${LBER_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set LDAP_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LDAP DEFAULT_MSG LDAP_INCLUDE_DIR LDAP_LIBRARY)

mark_as_advanced(LDAP_FOUND LDAP_INCLUDE_DIR LDAP_LIBRARY LBER_LIBRARY)

