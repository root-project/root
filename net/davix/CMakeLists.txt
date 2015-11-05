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


ROOT_GENERATE_DICTIONARY(G__DAVIX *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RDAVIX LINKDEF LinkDef.h DEPENDENCIES RIO Net)
ROOT_LINKER_LIBRARY(RDAVIX *.cxx G__DAVIX.cxx LIBRARIES ${DAVIX_LIBRARIES} ${OPENSSL_LIBRARIES} ${LIBXML2_LIBRARIES} ${SYSLIBS} DEPENDENCIES RIO Net)

if(builtin_davix)
  ROOT_ADD_BUILTIN_DEPENDENCIES(RDAVIX DAVIX)
endif()

if(builtin_openssl)
  ROOT_ADD_BUILTIN_DEPENDENCIES(RDAVIX OPENSSL)
endif()

ROOT_INSTALL_HEADERS()

