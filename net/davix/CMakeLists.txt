############################################################################
# CMakeLists.txt file for building ROOT net/davix package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(net/net)

include_directories(${DAVIX_INCLUDE_DIRS})

# Workaround for defects in davix 0.3.1: https://its.cern.ch/jira/browse/DMC-577
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-ignored-qualifiers)
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-unused-parameter)

ROOT_GENERATE_DICTIONARY(G__DAVIX *.h MODULE RDAVIX LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RDAVIX *.cxx G__DAVIX.cxx LIBRARIES ${DAVIX_LIBRARIES} ${OPENSSL_LIBRARIES} ${LIBXML2_LIBRARIES} ${SYSLIBS} DEPENDENCIES RIO Net)

if(builtin_davix)
  add_dependencies(G__DAVIX DAVIX)
  ROOT_ADD_BUILTIN_DEPENDENCIES(RDAVIX DAVIX)
endif()

ROOT_INSTALL_HEADERS()

