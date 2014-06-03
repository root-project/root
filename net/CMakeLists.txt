
add_subdirectory(net)
add_subdirectory(auth)  # special CMakeListst.txt

if(bonjour)
  add_subdirectory(bonjour)
endif()

if(krb5)
  add_subdirectory(krb5auth)
endif()

if(NOT WIN32 AND ssl)
  add_subdirectory(rpdutils)
endif()

if(xrootd)
  add_subdirectory(netx)
  if(alien)
    add_subdirectory(alien)
  endif()
endif()

if(monalisa)
  add_subdirectory(monalisa)
endif()

if(ldap)
   add_subdirectory(ldap)
endif()

if(globus)
   add_subdirectory(globusauth)
endif()

if(davix)
   add_subdirectory(davix)
endif()

if(netxng)
   add_subdirectory(netxng)
endif()

if(http)
   add_subdirectory(http)
endif()

