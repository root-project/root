ROOTTEST_ADD_TEST(delmacro
                  COMMAND ${CMAKE_COMMAND} -E remove -f "runConst_C.*"
                  FIXTURES_SETUP cling-const-delmacro-fixture)

ROOTTEST_ADD_TEST(run1
                  MACRO runConst.C
                  OUTREF good1.output
                  LABELS roottest regression cling
                  FIXTURES_REQUIRED cling-const-delmacro-fixture
                  FIXTURES_SETUP cling-const-run1-fixture)

ROOTTEST_ADD_TEST(run2
                  MACRO runConst.C+
                  OUTREF good2.output
                  LABELS roottest regression cling
                  FIXTURES_REQUIRED cling-const-run1-fixture)

ROOTTEST_ADD_TEST(constCharTest
                  MACRO constissue.C
                  LABELS roottest regression cling)
