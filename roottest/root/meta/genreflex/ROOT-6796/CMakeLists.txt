ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(attributesFromTypedef attributesFromTypedef.h SELECTION attributesFromTypedef_selection.xml)

ROOTTEST_ADD_TEST(attributesFromTypedef
                  MACRO execattributesFromTypedef.C
                  OUTREF execattributesFromTypedef.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
