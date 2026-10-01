ROOTTEST_ADD_TEST(cintrun
                  MACRO cintrun.C
                  OUTREF cintrun.ref-cling
                  DEPENDS t.h
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runreferenceuse
                  MACRO runreferenceUse.C
                  OUTREF referenceUse.ref
                  DEPENDS MyClassReferenceUse.C
                  LABELS roottest regression cling)

if(pyroot)
  ROOTTEST_ADD_TEST(testcint
                    MACRO testcint.py
                    OUTREF pythoncintrun.ref
                    OUTCNVCMD grep -v "just a comment"
                    DEPENDS t-build
                    LABELS roottest regression cling)
endif()
