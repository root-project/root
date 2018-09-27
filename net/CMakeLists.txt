add_subdirectory(net)
add_subdirectory(auth)

if(afs)
  add_subdirectory(afs)
endif()

if(bonjour)
  add_subdirectory(bonjour)
endif()

if(krb5)
  add_subdirectory(krb5auth)
endif()

if(NOT WIN32 AND ssl)
  add_subdirectory(rpdutils)
  add_subdirectory(rootd)
endif()

if(xrootd)
  add_subdirectory(netx)
endif()

if(alien)
  add_subdirectory(alien)
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
   add_subdirectory(httpsniff)
endif()

