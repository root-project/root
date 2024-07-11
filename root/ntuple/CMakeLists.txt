if(NOT ROOT_root7_FOUND)
  return()
endif()

ROOTTEST_ADD_TEST(test_Basics
                  MACRO basics.C
                  OUTREF test_basics.ref)

ROOT_ADD_GTEST(test_Merge merge.cxx LIBRARIES ROOT::ROOTNTuple)
