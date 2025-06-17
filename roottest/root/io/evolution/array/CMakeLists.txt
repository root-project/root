ROOTTEST_COMPILE_MACRO(MyClass.cxx
                       FIXTURES_SETUP root-io-evolution-array-myclass-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_myclass_load -e "(void)gROOT->ProcessLine(\".L MyClass.cxx+\")")
endif()

ROOTTEST_ADD_TEST(oldArrayObject
                  MACRO WriteOldObject.C
                  ROOTEXE_OPTS ${_myclass_load}
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-array-myclass-fixture
                  FIXTURES_SETUP root-io-evolution-array-oldArrayObject-fixture)

ROOTTEST_ADD_TEST(fixed2var
                  MACRO runfixed2var.C
                  OUTREF fixed2var.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-array-oldArrayObject-fixture)
