############################################################################
# CMakeLists.txt file for building ROOT net/rpdutils package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(net/net)
ROOT_USE_PACKAGE(net/auth)

file(GLOB rpdutilsrcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
list(REMOVE_ITEM rpdutilsrcs src/rpdpriv.cxx src/rpdconn.cxx)

if(GLOBUS_FOUND)
else()
  list(REMOVE_ITEM rpdutilsrcs src/globus.cxx)
endif()

#---Authentication methods-----------------------------------------------------------------
if(shadowpw)
  add_definitions(-DR__SHADOWPW)
endif()

#---static library needed for rootd and proofd --------------------------------------------
add_library(rpdutil STATIC ${rpdutilsrcs})

ROOT_LINKER_LIBRARY(SrvAuth rpdutils.cxx ssh.cxx LIBRARIES)


