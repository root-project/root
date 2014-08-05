# - Try to find Kerberos5
#  Check for libkrb5.a
#
#  KRB5_INCLUDE_DIR - where to find krb5.h, etc.
#  KRB5_INCLUDE_DIRS (not cached)
#  KRB5_LIBRARIES   - List of libraries when using Kerberos5
#  KRB5_INIT        - kinit command
#  KRB5_FOUND       - True if Kerberos 5 libraries found.

find_path(KRB5_INCLUDE_DIR NAMES krb5.h
  HINTS  $ENV{KRB5_DIR} ${KRB5_DIR}
  PATH_SUFFIXES include kerberos krb5)

set(KRB5_INCLUDE_DIRS ${KRB5_INCLUDE_DIR})

find_library(KRB5_LIBRARY NAMES krb5
  HINTS $ENV{KRB5_DIR} ${KRB5_DIR}
  PATH_SUFFIXES kerberos krb5)
set(KRB5_LIBRARIES ${KRB5_LIBRARY})

find_library(KRB5_MIT_LIBRARY NAMES k5crypto
  HINTS $ENV{KRB5_DIR} ${KRB5_DIR}
  PATH_SUFFIXES kerberos krb5)

if(KRB5_MIT_LIBRARY)
  set(KRB5_LIBRARIES ${KRB5_LIBRARIES} ${KRB5_MIT_LIBRARY})
endif()


find_program(KRB5_INIT NAMES kinit
  HINTS $ENV{KRB5_DIR}/bin ${KRB5_DIR}/bin
  PATH_SUFFIXES kerberos krb5)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args( KRB5 DEFAULT_MSG KRB5_LIBRARY KRB5_INCLUDE_DIR )

mark_as_advanced( KRB5_INCLUDE_DIR KRB5_MIT_LIBRARY KRB5_LIBRARY KRB5_INIT)



