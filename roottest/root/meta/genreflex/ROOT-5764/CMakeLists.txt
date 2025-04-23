if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)

    ROOTTEST_ADD_TEST(execLibLoad
                      MACRO execLibLoad.C
                      OUTREF execLibLoad.ref
                      DEPENDS ${GENERATE_REFLEX_TEST})
endif()
