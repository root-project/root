############################################################################
# CMakeLists.txt file for building ROOT main package
# @author Pere Mato, CERN
############################################################################

if(NOT WIN32)
  ROOT_EXECUTABLE(rootn.exe rmain.cxx LIBRARIES New Core MathCore Rint)
  ROOT_EXECUTABLE(roots.exe roots.cxx LIBRARIES Core MathCore)
  ROOT_EXECUTABLE(ssh2rpd ssh2rpd.cxx ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c )
endif()
if(zopfli)
  set(zopfli_library ${ZOPFLI_LIBRARIES})
endif()
ROOT_EXECUTABLE(root.exe rmain.cxx LIBRARIES Core Rint ${zopfli_library})
ROOT_EXECUTABLE(proofserv.exe pmain.cxx LIBRARIES Core MathCore)
ROOT_EXECUTABLE(hadd hadd.cxx LIBRARIES Core RIO Net Hist Graf Graf3d Gpad Tree Matrix MathCore Thread)
ROOT_EXECUTABLE(rootnb.exe nbmain.cxx LIBRARIES Core)

if(fortran AND CMAKE_Fortran_COMPILER)
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
    get_filename_component(python ${PYTHON_EXECUTABLE} NAME)
    configure_file(${rawUtilName} ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${utilName} @ONLY)

    install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${utilName}
                                 DESTINATION ${CMAKE_INSTALL_BINDIR} 
                                 RENAME ${utilName}
                                 PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                                             GROUP_EXECUTE GROUP_READ
                                             WORLD_EXECUTE WORLD_READ 
                                 COMPONENT applications)
  endforeach()

  install(FILES python/cmdLineUtils.py DESTINATION ${runtimedir})
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/cmdLineUtils.py )")
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -O -m py_compile \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${runtimedir}/cmdLineUtils.py )")
  configure_file(python/cmdLineUtils.py ${CMAKE_BINARY_DIR}/${runtimedir}/cmdLineUtils.py @ONLY)
endif()
