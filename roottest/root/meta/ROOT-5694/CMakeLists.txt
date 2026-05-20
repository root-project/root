if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(One One.h
                                        SELECTION One_selection.xml
                                        NO_ROOTMAP
                                        FIXTURES_SETUP root-meta-5694-One-fixture)

    ROOTTEST_GENERATE_REFLEX_DICTIONARY(Two Two.h
                                        SELECTION Two_selection.xml
                                        NO_ROOTMAP
                                        FIXTURES_SETUP root-meta-5694-Two-fixture)
    ROOTTEST_ADD_TEST(execLoadLibs
                      MACRO execLoadLibs.C
                      OUTREF execLoadLibs.oref
                      ERRREF execLoadLibs.eref
                      FIXTURES_REQUIRED root-meta-5694-One-fixture root-meta-5694-Two-fixture)
endif()
