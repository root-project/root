if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes
                                        classes.h
                                        SELECTION classes_selection.xml
                                        FIXTURES_SETUP root-meta-genreflex-5764-fixture)

    ROOTTEST_ADD_TEST(execLibLoad
                      MACRO execLibLoad.C
                      OUTREF execLibLoad.ref
                      FIXTURES_REQUIRED root-meta-genreflex-5764-fixture)
endif()
