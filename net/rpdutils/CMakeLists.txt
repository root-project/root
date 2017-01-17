############################################################################
# CMakeLists.txt file for building ROOT net/rpdutils package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_SOURCES(rpdutilsrcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
list(REMOVE_ITEM rpdutilsrcs src/rpdpriv.cxx src/rpdconn.cxx)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/res
  ${CMAKE_CURRENT_SOURCE_DIR}/../auth/res)

if(globus)
  add_definitions(-DR__GLBS)
  include_directories(${GLOBUS_INCLUDE_DIRS})
  if(GLOBUS_gsi_credential_FOUND)
    add_definitions(-DR__GLBS22)
  endif()
else()
  list(REMOVE_ITEM rpdutilsrcs src/globus.cxx)
endif()

if(MACOSX_SSL_DEPRECATED)
  set_source_files_properties(src/rpdutils.cxx COMPILE_FLAGS "-Wno-deprecated-declarations" )
endif()

#---Authentication methods-----------------------------------------------------------------
if(shadowpw)
  add_definitions(-DR__SHADOWPW)
endif()

if(ssl)
  include_directories(${OPENSSL_INCLUDE_DIR})
endif()

#---static library needed for rootd and proofd --------------------------------------------
add_library(rpdutil STATIC ${rpdutilsrcs})
add_dependencies(rpdutil move_headers)

ROOT_LINKER_LIBRARY(SrvAuth rpdutils.cxx ssh.cxx LIBRARIES rpdutil rsa ${CRYPTLIBS} ${GLOBUS_LIBRARIES} ${OPENSSL_LIBRARIES} DEPENDENCIES Net BUILTINS OPENSSL)

