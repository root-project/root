ROOTTEST_COMPILE_MACRO(Data.cxx
                       FIXTURES_SETUP root-treeformula-array-Data_cxx-fixture)

ROOTTEST_COMPILE_MACRO(fillTree.cxx
                       FIXTURES_SETUP root-treeformula-array-fillTree_cxx-fixture)

ROOTTEST_ADD_TEST(filltree
                  MACRO runFillTree.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L Data.cxx+\")"
                               -e "(void)gROOT->ProcessLine(\".L fillTree.cxx+\")"
                  FIXTURES_REQUIRED root-treeformula-array-Data_cxx-fixture
                                    root-treeformula-array-fillTree_cxx-fixture
                  FIXTURES_SETUP root-treeformula-array-filltree-fixture)

ROOTTEST_ADD_TEST(arraytest
                  MACRO runArrayTest.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L Data.cxx+\")"
                  OUTREF arrayDrawTest.good
                  FIXTURES_REQUIRED root-treeformula-array-Data_cxx-fixture
                                    root-treeformula-array-filltree-fixture)
