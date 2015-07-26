############################################################################
# CMakeLists.txt file for building ROOT main package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(core/rint)

if(NOT WIN32)
  ROOT_EXECUTABLE(rootn.exe rmain.cxx LIBRARIES New Core MathCore Rint)
  ROOT_EXECUTABLE(roots.exe roots.cxx LIBRARIES Core MathCore)
  ROOT_EXECUTABLE(ssh2rpd ssh2rpd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c )
endif()
ROOT_EXECUTABLE(root.exe rmain.cxx LIBRARIES Core Rint)
ROOT_EXECUTABLE(proofserv.exe pmain.cxx LIBRARIES Core MathCore)
ROOT_EXECUTABLE(hadd hadd.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread)

if(CMAKE_Fortran_COMPILER)
  ROOT_EXECUTABLE(g2root g2root.f LIBRARIES minicern)
  ROOT_EXECUTABLE(h2root h2root.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread minicern)
endif()

