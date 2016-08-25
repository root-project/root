############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(io/xml)

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

if(FASTCGI_INCLUDE_DIR AND FASTCGI_LIBRARY)
include_directories(${FASTCGI_INCLUDE_DIR})
endif()

if(NOT FASTCGI_INCLUDE_DIR OR NOT FASTCGI_LIBRARY)
set(FASTCGI_LIBRARY "")
add_definitions(-DHTTP_WITHOUT_FASTCGI)
endif()

add_definitions(-DUSE_WEBSOCKET)

# look for the realtime extensions library and use it if it exists
find_library(RT_LIBRARY rt)
if(RT_LIBRARY)
  set(RT_LIBRARIES ${RT_LIBRARY})
endif()

ROOT_GENERATE_DICTIONARY(G__HTTP *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RHTTP LINKDEF LinkDef.h DEPENDENCIES Thread Tree Hist Gpad Graf RIO XMLIO)
ROOT_LINKER_LIBRARY(RHTTP *.cxx civetweb/civetweb.c G__HTTP.cxx LIBRARIES ${SYSLIBS} ${ZLIB_LIB} ${RT_LIBRARIES} ${FASTCGI_LIBRARY} DEPENDENCIES Thread Tree Hist Gpad Graf RIO XMLIO)

ROOT_INSTALL_HEADERS()

