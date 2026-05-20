ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewInheritMulti
                             InheritMulti.h
                             LINKDEF InheritMultilinkdef.h
                             SOURCES InheritMulti.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new-InheritMulti-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewnamespace
                             namespace.h
                             LINKDEF namespacelinkdef.h
                             SOURCES namespace.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new-namespace-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewtemplate
                             template.h
                             LINKDEF templatelinkdef.h
                             SOURCES template.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new-template-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewnstemplate
                             nstemplate.h
                             LINKDEF nstemplatelinkdef.h
                             SOURCES nstemplate.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new-nstemplate-fixture)


ROOTTEST_ADD_TEST(run
                  MACRO Run.C
                  OUTREF newTest.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-newClassDef-new-InheritMulti-fixture
                                    root-io-newClassDef-new-namespace-fixture
                                    root-io-newClassDef-new-template-fixture
                                    root-io-newClassDef-new-nstemplate-fixture)
