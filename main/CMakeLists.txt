############################################################################
# CMakeLists.txt file for building ROOT main package
# @author Pere Mato, CERN
############################################################################

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

if(python)
  file(GLOB utils RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} python/root*)
  foreach(rawUtilName ${utils})
    get_filename_component(utilName ${rawUtilName} NAME)
    if(NOT WIN32)
      # We need the .py only on Windows
      string(REPLACE ".py" "" utilName ${utilName})
    endif()

    install(FILES ${rawUtilName} DESTINATION ${CMAKE_INSTALL_BINDIR}/${CMAKE_INSTALL_BINDIR} RENAME ${utilName})
    execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
                    COMMAND ${CMAKE_COMMAND} -E copy ${rawUtilName} ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${utilName})
  endforeach()

  install(FILES python/cmdLineUtils.py DESTINATION ${runtimedir})
  install(CODE "execute_process(COMMAND python -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/cmdLineUtils.py )")
  install(CODE "execute_process(COMMAND python -O -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/cmdLineUtils.py )")
  file(COPY python/cmdLineUtils.py DESTINATION ${CMAKE_BINARY_DIR}/${runtimedir})
endif()
