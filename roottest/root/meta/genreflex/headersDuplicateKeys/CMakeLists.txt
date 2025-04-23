ROOTTEST_GENERATE_REFLEX_DICTIONARY(a1 a1.h SELECTION a1_selection.xml)
set(dependency "${dependency};${GENERATE_REFLEX_TEST}")

ROOTTEST_GENERATE_REFLEX_DICTIONARY(a2 a2.h SELECTION a2_selection.xml)
set(dependency "${dependency};${GENERATE_REFLEX_TEST}")

ROOTTEST_GENERATE_REFLEX_DICTIONARY(a3 a3.h SELECTION a3_selection.xml)
set(dependency "${dependency};${GENERATE_REFLEX_TEST}")


ROOTTEST_ADD_TEST(headersDuplicateKeys
                  MACRO execduplicateHeaders.C
                  OUTREF execduplicateHeaders.ref
                  DEPENDS ${dependency})
