ROOTTEST_GENERATE_DICTIONARY(libPairs
                             pairtest.h
                             LINKDEF pairLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-6840-dict-fixture)

ROOTTEST_ADD_TEST(pairWriteFile
                  MACRO pairWrite.C
                  OUTREF pairWrite.ref
                  FIXTURES_REQUIRED root-meta-tclass-6840-dict-fixture
                  FIXTURES_SETUP root-meta-tclass-6840-pairWriteFile-fixture)

ROOTTEST_ADD_TEST(pairCalculation
                  MACRO execPair.C
                  OUTREF execPair.ref
                  FIXTURES_REQUIRED root-meta-tclass-6840-pairWriteFile-fixture)

ROOTTEST_GENERATE_EXECUTABLE(emulatePairs
                             emulatePairs.cpp
                             LIBRARIES Core RIO Tree
                             FIXTURES_SETUP root-meta-tclass-6840-emulatePairs-fixture)

ROOTTEST_ADD_TEST(emulatePairsTest
                  EXEC ./emulatePairs
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:pointervalue:g -e s:->[0-9a-fA-F]*:pointervalue:g
                  OUTREF emulatePairs.ref
                  FIXTURES_REQUIRED root-meta-tclass-6840-dict-fixture root-meta-tclass-6840-emulatePairs-fixture)
