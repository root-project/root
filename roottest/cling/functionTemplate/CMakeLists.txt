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

if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(testcint
                    MACRO testcint.py
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ t.h+
                    OUTREF pythoncintrun.ref
                    OUTCNVCMD grep -v "just a comment"
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    LABELS roottest regression cling)

  if(ROOT_cintex_FOUND)
    ROOTTEST_ADD_TEST(testreflex
                      MACRO testreflex.py
                      OUTREF pythonreflexrun.ref
                      OUTCNVCMD grep -v "just a comment"
                      LABELS roottest regression cling)
  endif()
endif()
