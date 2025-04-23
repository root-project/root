if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(One One.h
                                        SELECTION One_selection.xml
                                        NO_ROOTMAP
                                        FIXTURES_SETUP execLoadLibs_fixture)

    ROOTTEST_GENERATE_REFLEX_DICTIONARY(Two Two.h
                                        SELECTION Two_selection.xml
                                        NO_ROOTMAP
                                        FIXTURES_SETUP execLoadLibs_fixture)
    ROOTTEST_ADD_TEST(execLoadLibs
                      MACRO execLoadLibs.C
                      OUTREF execLoadLibs.oref
                      ERRREF execLoadLibs.eref
                      FIXTURES_REQUIRED execLoadLibs_fixture)
endif()
