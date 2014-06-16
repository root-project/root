ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-cintrun
                  MACRO cintrun.C
                  OUTREF cintrun.ref-cling
                  DEPENDS t.h
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-runreferenceuse
                  MACRO runreferenceUse.C
                  OUTREF referenceUse.ref
                  DEPENDS MyClassReferenceUse.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-testcint
                  MACRO testcint.py
                  PRECMD root -b -q -l -e .L\ t.h+
                  OUTREF pythoncintrun.ref
                  OUTCNVCMD grep -v "just a comment"
                  LABELS roottest regression cling)

if(ROOT_cintex_FOUND)
  ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-testreflex
                    MACRO testreflex.py
                    OUTREF pythonreflexrun.ref
                    OUTCNVCMD grep -v "just a comment"
                    LABELS roottest regression cling)
endif()
