ROOTTEST_ADD_TEST(roottest-root-html-runClassInModule
                  MACRO runClassInModule.C
                  WILLFAIL
                  OUTREF ClassInModule.ref)

ROOTTEST_ADD_TEST(roottest-root-html-runFindSource
                  MACRO runFindSource.C
                  WILLFAIL
                  OUTREF FindSource.ref)

ROOTTEST_ADD_TEST(roottest-root-html-runMakeIndex
                  MACRO runMakeIndex.C
                  OUTCNV MakeIndex_convert.sh
                  OUTREF MakeIndex.oref
                  ERRREF MakeIndex.eref)
