ROOTTEST_GENERATE_REFLEX_DICTIONARY(myClass myClass.h SELECTION myClass_selection.xml)

set(dependencies ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Odin Odin.h SELECTION Odin_selection.xml)

set(dependencies ${dependencies} ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(execStaticConst
                  MACRO execStaticConst.C
                  OUTREF execStaticConst.ref
                  DEPENDS ${dependencies}
                  WILLFAIL
                  LABELS roottest regression cling)
