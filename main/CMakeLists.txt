############################################################################
# CMakeLists.txt file for building ROOT main package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(core/rint)

configure_file(${CMAKE_SOURCE_DIR}/config/proofserv.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/proofserv.sh @ONLY)
install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/proofserv.sh DESTINATION bin)
configure_file(${PROJECT_SOURCE_DIR}/config/roots.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots.sh @ONLY)
install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots.sh DESTINATION bin)

if(NOT WIN32)
  ROOT_EXECUTABLE(rootn.exe rmain.cxx LIBRARIES New Core Cint MathCore Rint)
  ROOT_EXECUTABLE(roots.exe roots.cxx LIBRARIES Core Cint MathCore)
  ROOT_EXECUTABLE(ssh2rpd ssh2rpd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c )
endif()
ROOT_EXECUTABLE(root.exe rmain.cxx LIBRARIES Core Cint MathCore Rint)
ROOT_EXECUTABLE(proofserv.exe pmain.cxx LIBRARIES Core Cint MathCore)
ROOT_EXECUTABLE(hadd hadd.cxx LIBRARIES Core Cint RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread)

if(CMAKE_Fortran_COMPILER)
  ROOT_EXECUTABLE(g2root g2root.f LIBRARIES minicern)
  #ROOT_EXECUTABLE(h2root h2root.cxx LIBRARIES Core Cint RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread minicern)
endif()

