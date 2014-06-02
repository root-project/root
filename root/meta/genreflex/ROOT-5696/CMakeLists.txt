ROOTTEST_GENERATE_REFLEX_DICTIONARY(warnMe warnMe.h SELECTION warnMe_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5696-exec5596
                  MACRO exec5596.C
                  OUTREF exec5596.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
