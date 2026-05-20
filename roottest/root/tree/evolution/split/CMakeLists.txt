# LABELS longtest

ROOTTEST_ADD_TEST(CreateFileFullObjRule
                  MACRO execCreateFileFullObjRule.cxx+
                  OUTREF execCreateFileFullObjRule.ref
                  FIXTURES_SETUP root-tree-evolution-split-CreateFileFullObjRule-fixture)

ROOTTEST_ADD_TEST(ReadFileFullObjRule
                  MACRO execReadFileFullObjRule.cxx+
                  OUTREF execReadFileFullObjRule.ref
                  FIXTURES_REQUIRED root-tree-evolution-split-CreateFileFullObjRule-fixture)

ROOTTEST_ADD_TEST(makeclass
                  MACRO execmakeclass.C
                  OUTREF execmakeclass.ref)

ROOTTEST_ADD_TEST(missingsplit_write
                  MACRO missingsplit_write.C+
                  FIXTURES_SETUP root-tree-evolution-split-missingsplit_write-fixture)

ROOTTEST_COMPILE_MACRO(missingsplit_read.C
                       FIXTURES_SETUP root-tree-evolution-split-missingsplit_read-fixture)

ROOTTEST_ADD_TEST(missingsplit
                  MACRO runmissingsplit.C
                  OUTREF missingsplit.ref
                  FIXTURES_REQUIRED root-tree-evolution-split-missingsplit_write-fixture
                                    root-tree-evolution-split-missingsplit_read-fixture)
