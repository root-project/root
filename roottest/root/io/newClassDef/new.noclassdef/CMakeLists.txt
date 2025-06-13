ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewNoDefnamespace
                             namespace.h
                             LINKDEF namespacelinkdef.h
                             SOURCES namespace.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new_noclassdef-namespace-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewNoDeftemplate
                             template.h
                             LINKDEF templatelinkdef.h
                             SOURCES template.C
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new_noclassdef-template-fixture)

ROOTTEST_GENERATE_DICTIONARY(libIoNewClassNewNoDefnstemplate
                             nstemplate.h
                             LINKDEF nstemplatelinkdef.h
                             SOURCES nstemplate.C
                             COMPILE_OPTIONS ${_opt}
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-new_noclassdef-nstemplate-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO Run.C
                  FIXTURES_REQUIRED root-io-newClassDef-new_noclassdef-namespace-fixture
                                    root-io-newClassDef-new_noclassdef-template-fixture
                                    root-io-newClassDef-new_noclassdef-nstemplate-fixture)
