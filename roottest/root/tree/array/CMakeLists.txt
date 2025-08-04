ROOTTEST_ADD_TEST(StdArray
                  MACRO  execStdArray.C
                  OUTREF execStdArray.ref)

ROOTTEST_COMPILE_MACRO(TestObj.cpp
                       FIXTURES_SETUP root-tree-array-TestObj-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_load_class -e "(void)gROOT->ProcessLine(\".L TestObj.cpp+\")")
endif()

ROOTTEST_ADD_TEST(objArrayTest
                  ROOTEXE_OPTS ${_load_class}
                  MACRO runobjArrayTest.C
                  OUTREF objArrayTest.ref
                  FIXTURES_REQUIRED root-tree-array-TestObj-fixture)

ROOTTEST_ADD_TEST(objArrayTest2
                  MACRO run2.C
                  OUTREF objArrayTest2.ref
                  FIXTURES_REQUIRED root-tree-array-TestObj-fixture root-tree-array-save-fixture)
