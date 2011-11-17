add_subdirectory(net)
add_subdirectory(auth)  # special CMakeListst.txt

if(bonjour)
  add_subdirectory(bonjour)
endif()

if(krb5)
  add_subdirectory(krb5auth)
endif()

if(xrootd)
  add_subdirectory(netx)
  if(NOT WIN32)
    add_subdirectory(rpdutils)
  endif()
endif()

if(alien)
  add_subdirectory(alien)
endif()

if(monalisa)
  add_subdirectory(monalisa)
endif()



