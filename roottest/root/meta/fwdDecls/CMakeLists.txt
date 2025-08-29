# ROOTTEST_GENERATE_REFLEX_DICTIONARY(clingFwdDeclExample clingFwdDeclExample.h SELECTION clingFwdDeclExample_selection.xml NO_ROOTMAP)

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(clingFwdDeclExample
                      COPY_TO_BUILDDIR clingFwdDeclExample.h clingFwdDeclExample_selection.xml A.h checkDict.sh
                      COMMAND ./checkDict.sh
                      OUTREF clingFwdDeclExample.ref)
endif()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(fwdDeclarations
                                    fwdDeclarations.h
                                    SELECTION fwdDeclarations_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-fwdDecls-fwdDeclarations-fixture)

ROOTTEST_ADD_TEST(fwdDeclarations
                  MACRO execfwdDeclarations.C
                  OUTREF execfwdDeclarations.ref
                  ${WILLFAIL_ON_WIN32}
                  FIXTURES_REQUIRED root-meta-fwdDecls-fwdDeclarations-fixture)
