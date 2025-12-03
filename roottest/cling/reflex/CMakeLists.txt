ROOTTEST_GENERATE_REFLEX_DICTIONARY(classVersion
                                    classVersion.h
                                    SELECTION classVersion_selection.xml
                                    FIXTURES_SETUP cling-reflex-classVersion-fixture)

# test fail on Windows - with warning:  no dictionary for class WithClassVersion is available
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(classVersion_rflx
                    MACRO classVersion_test.C
                    ROOTEXE_OPTS -e "(void)gSystem->Load(\"libclassVersion_dictrflx\")"
                    OUTREF classVersion_rflx.ref
                    FIXTURES_REQUIRED cling-reflex-classVersion-fixture)
endif()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(properties
                                    properties.h
                                    SELECTION properties_selection.xml
                                    FIXTURES_SETUP cling-reflex-properties-fixture)

ROOTTEST_ADD_TEST(properties_rflx
                  MACRO properties_test.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"libproperties_dictrflx\")"
                  OUTREF properties_rflx.ref
                  FIXTURES_REQUIRED cling-reflex-properties-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(iotype
                                    iotype.h
                                    SELECTION iotype_selection.xml
                                    FIXTURES_SETUP cling-reflex-iotype-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(iotype32
                                    iotype32.h
                                    SELECTION iotype32_selection.xml
                                    FIXTURES_SETUP cling-reflex-iotype32-fixture)

ROOTTEST_ADD_TEST(iotype_rflx
                  MACRO iotype_test.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"libiotype_dictrflx\")"
                               -e "(void)gSystem->Load(\"libiotype32_dictrflx\")"
                  OUTREF iotype_rflx.ref
                  FIXTURES_REQUIRED cling-reflex-iotype-fixture cling-reflex-iotype32-fixture)
