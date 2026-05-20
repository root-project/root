ROOTTEST_GENERATE_DICTIONARY(libIoNewClassDefNoClassDef
                             RootData.h
                             RootPCellID.h
                             RootCaloHit.h
                             LINKDEF LinkDef.h
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newClassDef-noclassdef-dict-fixture)

ROOTTEST_GENERATE_EXECUTABLE(driver
                             main.cc
                             LIBRARIES Core RIO libIoNewClassDefNoClassDeflibgen
                             FIXTURES_REQUIRED root-io-newClassDef-noclassdef-dict-fixture
                             FIXTURES_SETUP root-io-newClassDef-noclassdef-driver-fixture)

ROOTTEST_ADD_TEST(noclassdef
                  COMMAND ./driver
                  FIXTURES_REQUIRED root-io-newClassDef-noclassdef-driver-fixture
                  OUTREF good.linux.64.output
                  OUTCNVCMD sed -e s,Dumping[[:space:]]object[[:space:]]at:[[:space:]]0x[0-9a-f]*,Dumping\\\ object,g
                                -e s,TStreamerInfo\\\:\\\:,,g
                                -e s,type[[:space:]][0-9]*RootPCnodict,type\\\ class\\\ RootPCnodict,g)
