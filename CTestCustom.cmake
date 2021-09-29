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

if(CTEST_BUILD_NAME MATCHES clang39)  # ABI mismatch between our clang and external clang
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE
              tutorial-tmva-TMVAClassification
              tutorial-tmva-TMVAMulticlass
              tutorial-tmva-TMVAClassificationApplication
              tutorial-tmva-TMVAClassificationCategoryApplication
              tutorial-tmva-TMVARegressionApplication
              tutorial-tmva-TMVAMulticlassApplication
              tutorial-tmva-TMVARegression
              roottest-cling-stl-map-stringMap
              roottest-root-meta-genreflex-TClass-execbasic
              roottest-cling-stl-map-badstringMap)
endif()
