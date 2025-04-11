ROOTTEST_GENERATE_DICTIONARY(libPairs
                  pairtest.h
                  LINKDEF pairLinkDef.h
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_ADD_TEST(pairCalculation
                  MACRO execPair.C
                  OUTREF execPair.ref
                  DEPENDS libPairs-build pairWriteFile)

ROOTTEST_ADD_TEST(pairWriteFile
                  MACRO pairWrite.C
                  OUTREF pairWrite.ref
                  DEPENDS libPairs-build)


ROOTTEST_GENERATE_EXECUTABLE(emulatePairs emulatePairs.cpp LIBRARIES Core RIO Tree)

ROOTTEST_ADD_TEST(emulatePairsTest
                  EXEC ./emulatePairs
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:pointervalue:g -e s:->[0-9a-fA-F]*:pointervalue:g
                  OUTREF emulatePairs.ref
                  DEPENDS libPairs-build emulatePairs-build)
