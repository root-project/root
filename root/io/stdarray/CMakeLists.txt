ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(modelReadDict2TFile
                  MACRO  modelReadDict2.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict2.ref
                  DEPENDS roottest-root-io-stdarray-aclicModelWrite)

# ROOTTEST_ADD_TEST(modelReadDict2TXMLFile
#                   MACRO  modelReadDict2.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadDict2.ref
#                   DEPENDS roottest-root-io-stdarray-aclicModelWrite)

# The dependency is there not to compile the macros simultaneously
ROOTTEST_ADD_TEST(modelReadDictTFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict.ref
                  DEPENDS roottest-root-io-stdarray-modelReadDictTXMLFile)

ROOTTEST_ADD_TEST(modelReadDictTXMLFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.xml\""
                  OUTREF modelReadDict.ref
                  DEPENDS roottest-root-io-stdarray-aclicModelWrite)

ROOTTEST_ADD_TEST(modelReadNoDictTFile
                  MACRO  modelReadNoDict.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadNoDict.ref
                  DEPENDS roottest-root-io-stdarray-aclicModelWrite)

# ROOTTEST_ADD_TEST(modelReadNoDictTXMLFile
#                   MACRO  modelReadNoDict.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadNoDict.ref
#                   DEPENDS roottest-root-io-stdarray-aclicModelWrite)

ROOTTEST_ADD_TEST(aclicModelWrite
                  MACRO  aclicModelWrite.C+)

ROOTTEST_ADD_TEST(aclic03
                  MACRO  aclic03.C+
                  OUTREF aclic03.ref)

ROOTTEST_ADD_TEST(aclic01
                  MACRO  aclic01.C+
                  OUTREF aclic01.ref)

ROOTTEST_ADD_TEST(aclic00
                  MACRO  aclic00.C+
                  OUTREF aclic00.ref)
