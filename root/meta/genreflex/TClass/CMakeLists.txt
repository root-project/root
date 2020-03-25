ROOTTEST_GENERATE_REFLEX_DICTIONARY(basic_allClasses basic_allClasses.h SELECTION basic_allClasses_selection.xml)

if(NOT 32BIT)
   ROOTTEST_ADD_TEST(execbasic
                     MACRO execbasic.C
                     OUTREF execbasic.ref
                     DEPENDS ${GENERATE_REFLEX_TEST})
endif()
