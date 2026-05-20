ROOTTEST_COMPILE_MACRO(TestClass.cc
                       FIXTURES_SETUP root-io-references-actions-fixture)

if(ClingWorkAroundMissingSmartInclude OR ClingWorkAroundMissingDynamicScope)
  set(_load_class -e "(void)gROOT->ProcessLine(\".L TestClass.cc+\")")
endif()

ROOTTEST_ADD_TEST(test
                  ROOTEXE_OPTS ${_load_class}
                  MACRO test.C
                  OUTREF test.ref
                  FIXTURES_REQUIRED root-io-references-actions-fixture
                  FIXTURES_SETUP root-io-references-actions-test-fixture)

ROOTTEST_ADD_TEST(test1
                  ROOTEXE_OPTS ${_load_class}
                  MACRO test1.C
                  OUTREF test1.ref
                  FIXTURES_REQUIRED root-io-references-actions-test-fixture
                  FIXTURES_SETUP root-io-references-actions-test1-fixture)

ROOTTEST_ADD_TEST(test2
                  ROOTEXE_OPTS ${_load_class}
                  MACRO test2.C
                  OUTREF test2.ref
                  FIXTURES_REQUIRED root-io-references-actions-test1-fixture
                  FIXTURES_SETUP root-io-references-actions-test2-fixture)

ROOTTEST_ADD_TEST(test3
                  ROOTEXE_OPTS ${_load_class}
                  MACRO test3.C
                  OUTREF test3.ref
                  FIXTURES_REQUIRED root-io-references-actions-test2-fixture)
