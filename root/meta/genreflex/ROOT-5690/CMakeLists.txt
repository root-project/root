ROOTTEST_GENERATE_REFLEX_DICTIONARY(FastAllocVector FastAllocVector.h SELECTION FastAllocVector_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5690-execdummy
                  MACRO execdummy.C
                  OUTREF execdummy.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
