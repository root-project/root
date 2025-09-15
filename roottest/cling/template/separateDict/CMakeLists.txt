ROOTTEST_GENERATE_DICTIONARY(libmasterSeparate
                             MyClass.hh
                             SOURCES MyClass.C
                             LINKDEF MasterLinkDef.hh
                             FIXTURES_SETUP cling-template-separateDict-master-fixture)

ROOTTEST_GENERATE_DICTIONARY(libslave1Separate
                             MyClass.hh
                             LINKDEF Slave1LinkDef.hh
                             FIXTURES_SETUP cling-template-separateDict-slave1-fixture)

ROOTTEST_GENERATE_DICTIONARY(libslave2Separate
                             MyClass.hh
                             LINKDEF Slave2LinkDef.hh
                             FIXTURES_SETUP cling-template-separateDict-slave2-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO run.C
                  OUTREF_CINTSPECIFIC sepDictLibs.ref
                  OUTCNVCMD sed -e "s?FILE:[A-Za-z]:.*[/\]?FILE:?" -e "s?FILE:/.*/?FILE:?"  -e "/List of member variables/,/List of member functions/d" -e "s?size=0x[0-9a-fA-F]*?size=n/a?"
                  FIXTURES_REQUIRED cling-template-separateDict-master-fixture
                                    cling-template-separateDict-slave1-fixture
                                    cling-template-separateDict-slave2-fixture)