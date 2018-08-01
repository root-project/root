add_root_subdirectory(net)
add_root_subdirectory(auth)  # special CMakeListst.txt

if(bonjour)
  add_root_subdirectory(bonjour)
endif()

if(krb5)
  add_root_subdirectory(krb5auth)
endif()

if(NOT WIN32 AND ssl)
  add_root_subdirectory(rpdutils)
  add_root_subdirectory(rootd)
endif()

if(xrootd)
  add_root_subdirectory(netx)
endif()

if(alien)
  add_root_subdirectory(alien)
endif()

if(monalisa)
  add_root_subdirectory(monalisa)
endif()

if(ldap)
   add_root_subdirectory(ldap)
endif()

if(globus)
   add_root_subdirectory(globusauth)
endif()

if(davix)
   add_root_subdirectory(davix)
endif()

if(netxng)
   add_root_subdirectory(netxng)
endif()

if(http)
   add_root_subdirectory(http)
   add_root_subdirectory(httpsniff)
endif()

