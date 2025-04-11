#-------------------------------------------------------------------------------
#
# CTestCustom.cmake
#
# This file enables customization of CTest.
#
#-------------------------------------------------------------------------------

# Specify tests that will be ignored.

list(APPEND CTEST_CUSTOM_TESTS_IGNORE
            roottest-cling-parsing-semicolon)

if(CTEST_BUILD_NAME MATCHES slc6|centos7)
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE
              roottest-root-meta-loadAllLibs-LoadAllLibs
              roottest-root-meta-loadAllLibs-LoadAllLibsAZ
              roottest-root-meta-loadAllLibs-LoadAllLibsZA
              roottest-root-html-runMakeIndex
              roottest-root-multicore-fork)
endif()

if(WIN32)
  # driveTabCom.py: `import pty` is not supported on Windows
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE
              roottest-root-rint-TabCom
              roottest-root-rint-BackslashNewline)
endif()

if(CTEST_BUILD_NAME MATCHES fst)
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE roottest-python-JsMVA-NewMethods)
endif()
