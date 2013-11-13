#--- ROOT_SOURCE_DIR is required-----------------------------------------------

if(ROOT_cintex_FOUND) 

  set(CINTEXTEST_DIR ${ROOT_SOURCE_DIR}/cint/cintex/test)
  REFLEX_GENERATE_DICTIONARY(CintexTest ${CINTEXTEST_DIR}/dict/CintexTest.h SELECTION ${CINTEXTEST_DIR}/dict/selection.xml OPTIONS --comments )
  ROOT_MODULE_LIBRARY(CintexTestDict CintexTest_dict.cpp LIBRARIES Reflex)

  ROOT_ADD_TEST(cint-cintex-build 
                SOURCE_DIR ${CMAKE_SOURCE_DIR}
                BINARY_DIR ${CMAKE_BINARY_DIR}
                BUILD  CintexTestDict)                                              
  ROOT_ADD_TEST(cint-cintex-cintex 
                COMMAND root -b -q -l ${CINTEXTEST_DIR}/test_Cintex.C
                FAILREGEX "Test failed"
                DEPENDS cint-cintex-build)
  ROOT_ADD_TEST(cint-cintex-persistency 
                COMMAND root -b -q -l ${CINTEXTEST_DIR}/test_Persistency.C
                FAILREGEX "Test failed"
                DEPENDS cint-cintex-build)
  ROOT_ADD_TEST(cint-cintex-pycintex
                COMMAND  ${python_cmd} ${CINTEXTEST_DIR}/test_PyCintex_basics.py - -b
                FAILREGEX "Test failed"
                DEPENDS cint-cintex-build)

endif()
                                
