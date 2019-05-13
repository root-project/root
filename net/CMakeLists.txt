add_subdirectory(net)
add_subdirectory(auth)

if(NOT WIN32 AND ssl)
  add_subdirectory(rpdutils)
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

