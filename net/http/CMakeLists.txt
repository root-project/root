# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
# @author Sergey Linev, GSI
############################################################################

if(NOT FASTCGI_FOUND)
  set(FASTCGI_LIBRARY "")
endif()

# look for the realtime extensions library and use it if it exists
find_library(RT_LIBRARY rt)
if(RT_LIBRARY)
  set(RT_LIBRARIES ${RT_LIBRARY})
endif()

if(builtin_civetweb)
   set(_civetweb_src civetweb/civetweb.c)
   set(_civetweb_libs ${RT_LIBRARIES})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTP
  HEADERS
    THttpCallArg.h
    THttpEngine.h
    THttpServer.h
    THttpWSHandler.h
    TRootSniffer.h
    TRootSnifferStore.h
  SOURCES
    src/TCivetweb.cxx
    src/TCivetweb.h
    src/TFastCgi.cxx
    src/TFastCgi.h
    src/THttpCallArg.cxx
    src/THttpEngine.cxx
    src/THttpLongPollEngine.cxx
    src/THttpLongPollEngine.h
    src/THttpServer.cxx
    src/THttpWSEngine.cxx
    src/THttpWSEngine.h
    src/THttpWSHandler.cxx
    src/TRootSniffer.cxx
    src/TRootSnifferStore.cxx
    ${_civetweb_src}
  LIBRARIES
    ZLIB::ZLIB
    ${_civetweb_libs}
    ${FASTCGI_LIBRARY}
    ${CMAKE_DL_LIBS}
  DEPENDENCIES
    RIO
    Thread
)

if(builtin_civetweb)
   target_include_directories(RHTTP PRIVATE civetweb)

  if(ssl)
    target_include_directories(RHTTP SYSTEM PRIVATE ${OPENSSL_INCLUDE_DIR})
  endif()

  target_compile_definitions(RHTTP PRIVATE -DUSE_WEBSOCKET)

  if(NOT MSVC)
    target_compile_definitions(RHTTP PRIVATE -DUSE_X_DOM_SOCKET)
  endif()

  if(ssl)
    if(OPENSSL_VERSION)
      string(REPLACE "." ";" lst ${OPENSSL_VERSION})
      list(GET lst 0 ssl_major)
      list(GET lst 1 ssl_minor)
    endif()

    if((${ssl_major} EQUAL "1") AND (${ssl_minor} EQUAL "1"))
      MESSAGE(STATUS "Use SSL API VERSION 1.1 for civetweb")
      target_compile_definitions(RHTTP PUBLIC -DOPENSSL_API_1_1)
      set(link_ssl ON)
    elseif((${ssl_major} EQUAL "3"))
      MESSAGE(STATUS "Use SSL API VERSION 3.${ssl_minor} for civetweb")
      target_compile_definitions(RHTTP PUBLIC -DOPENSSL_API_3_0)
      set(link_ssl ON)
    elseif((${ssl_major} EQUAL "1") AND (${ssl_minor} EQUAL "0"))
      MESSAGE(STATUS "Use SSL API VERSION 1.0 for civetweb")
      target_compile_definitions(RHTTP PUBLIC -DOPENSSL_API_1_0)
      set(link_ssl ON)
    else()
      MESSAGE(WARNING "Not able to recognize SSL version ${OPENSSL_VERSION}, disable SSL")
      target_compile_definitions(RHTTP PUBLIC -DNO_SSL)
    endif()
    if(link_ssl)
      target_compile_definitions(RHTTP PUBLIC -DNO_SSL_DL)
      target_link_libraries(RHTTP PRIVATE ${OPENSSL_LIBRARIES})
    endif()
  else()
    target_compile_definitions(RHTTP PUBLIC -DNO_SSL)
  endif()

else()

  target_include_directories(RHTTP SYSTEM PRIVATE ${civetweb_INCLUDE_DIR})
  target_link_libraries(RHTTP PRIVATE ${civetweb_LIBRARIES})
  # workaround for R__memcompress failure with external civetweb
  target_compile_definitions(RHTTP PRIVATE -D_EXTERNAL_CIVETWEB)

endif(builtin_civetweb)

if(FASTCGI_FOUND)
  target_include_directories(RHTTP PRIVATE ${FASTCGI_INCLUDE_DIR})
else()
  target_compile_definitions(RHTTP PUBLIC -DHTTP_WITHOUT_FASTCGI)
endif()
