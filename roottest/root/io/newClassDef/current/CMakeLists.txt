ROOTTEST_GENERATE_DICTIONARY(libIoNewClassInheritMulti
                             InheritMulti.h
                             LINKDEF InheritMultilinkdef.h
                             SOURCES InheritMulti.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-current-InheritMulti-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassnamespace
                             namespace.h
                             LINKDEF namespacelinkdef.h
                             SOURCES namespace.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-current-namespace-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClasstemplate
                             template.h
                             LINKDEF templatelinkdef.h
                             SOURCES template.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-current-template-fixture)

ROOTTEST_COMPILE_MACRO(array.cxx
                       FIXTURES_SETUP root-io-newClassDef-current-array-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO Run.C
                  OUTREF current_test.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-newClassDef-current-InheritMulti-fixture
                                    root-io-newClassDef-current-namespace-fixture
                                    root-io-newClassDef-current-template-fixture
                                    root-io-newClassDef-current-array-fixture)
