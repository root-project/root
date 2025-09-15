ROOTTEST_GENERATE_DICTIONARY(libmasterSeparateNS
                             Master.hh
                             SOURCES Master.C
                             LINKDEF MasterLinkDef.hh
                             FIXTURES_SETUP cling-template-separateDictNamespace-master-fixture)

ROOTTEST_GENERATE_DICTIONARY(libslave1SeparateNS
                             Slave1.hh
                             SOURCES Slave1.C
                             LINKDEF Slave1LinkDef.hh
                             FIXTURES_SETUP cling-template-separateDictNamespace-slave1-fixture)

ROOTTEST_GENERATE_DICTIONARY(libslave2SeparateNS
                             Slave2.hh
                             SOURCES Slave2.C
                             LINKDEF Slave2LinkDef.hh
                             FIXTURES_SETUP cling-template-separateDictNamespace-slave2-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO run.C
                  OUTREF_CINTSPECIFIC sepDictLibs.ref
                  OUTCNVCMD sed -e "s?FILE:[A-Za-z]:.*[/\]?FILE:?" -e "s?FILE:/.*/?FILE:?"  -e "/List of member variables/,/List of member functions/d" -e "s?size=0x[0-9a-fA-F]*?size=n/a?"
                  FIXTURES_REQUIRED cling-template-separateDictNamespace-master-fixture
                                    cling-template-separateDictNamespace-slave1-fixture
                                    cling-template-separateDictNamespace-slave2-fixture)