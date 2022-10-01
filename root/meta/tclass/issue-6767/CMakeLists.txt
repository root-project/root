ROOTTEST_GENERATE_DICTIONARY(libRepro
                  repro.h
                  LINKDEF reproLinkDef.h
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_ADD_TEST(reproducer
                  MACRO repro.C
                  OUTREF repro.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST})
