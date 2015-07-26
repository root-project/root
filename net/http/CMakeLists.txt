############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(io/xml)

ROOT_GLOB_HEADERS(headers inc/*.h)
ROOT_GLOB_SOURCES(sources src/*.cxx civetweb/civetweb.c)

find_path(FASTCGI_INCLUDE_DIR fcgiapp.h
  $ENV{FASTCGI_DIR}/include
  /usr/local/include
  /usr/include/fastcgi
  /usr/local/include/fastcgi
  /opt/fastcgi/include
  DOC "Specify the directory containing fcgiapp.h"
)

find_library(FASTCGI_LIBRARY NAMES fcgi PATHS
  $ENV{FASTCGI_DIR}/lib
  /usr/local/fastcgi/lib
  /usr/local/lib
  /usr/lib/fastcgi
  /usr/local/lib/fastcgi
  /usr/fastcgi/lib /usr/lib
  /usr/fastcgi /usr/local/fastcgi
  /opt/fastcgi /opt/fastcgi/lib
  DOC "Specify the FastCGI library here."
)

mark_as_advanced(FASTCGI_INCLUDE_DIR FASTCGI_LIBRARY)

if(FASTCGI_INCLUDE_DIR AND FASTCGI_LIBRARY)
  include_directories(${FASTCGI_INCLUDE_DIR})
endif()

if(ssl)
  include_directories(${OPENSSL_INCLUDE_DIR})
else()
  set(OPENSSL_LIBRARIES)
endif()

if(NOT FASTCGI_INCLUDE_DIR OR NOT FASTCGI_LIBRARY)
  set(FASTCGI_LIBRARY "")
  add_definitions(-DHTTP_WITHOUT_FASTCGI)
endif()

ROOT_GENERATE_DICTIONARY(G__RHTTP ${headers} MODULE RHTTP LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RHTTP ${sources} G__RHTTP.cxx LIBRARIES ${SYSLIBS} ${ZLIB_LIBRARY} ${FASTCGI_LIBRARY} ${OPENSSL_LIBRARIES} DEPENDENCIES Thread Tree Hist Gpad Graf RIO XMLIO)


ROOT_INSTALL_HEADERS()
