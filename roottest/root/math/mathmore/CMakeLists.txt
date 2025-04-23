if(ROOT_mathmore_FOUND)
ROOTTEST_ADD_TEST(testLoadLibMathMore
                  OUTREF testLoadLibMathMore.ref	
                  MACRO testLoadLibMathMore.C)
endif()
