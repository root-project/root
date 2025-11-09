ROOTTEST_COMPILE_MACRO(myclass1.cc
                       FIXTURES_SETUP root-treeformula-evolution-myclass1-fixture)

ROOTTEST_COMPILE_MACRO(myclass2.cc
                       FIXTURES_SETUP root-treeformula-evolution-myclass2-fixture)

ROOTTEST_ADD_TEST(writeclass
                  MACRO writeclass.cc
                  FIXTURES_REQUIRED root-treeformula-evolution-myclass1-fixture
                  FIXTURES_SETUP root-treeformula-evolution-writeclass-fixture)

ROOTTEST_ADD_TEST(readclass
                  MACRO readclass.cc
                  OUTREF treetest.ref
                  FIXTURES_REQUIRED root-treeformula-evolution-myclass2-fixture
                                    root-treeformula-evolution-writeclass-fixture)
