# More coverage for ROOT-8736
ROOTTEST_ADD_TEST(arrTemplMemberWrite
                  MACRO testArrOfTemplateInst.C+
                  OUTREF testArrOfTemplateInst.ref)

ROOTTEST_ADD_TEST(colWiseWrite_1
                  MACRO colWiseWrite_1.C+
                  FIXTURES_SETUP root-io-stdarray-colWiseWrite_1-fixture)

ROOTTEST_ADD_TEST(colWiseRead_11
                  MACRO colWiseRead_11.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_1-fixture)

ROOTTEST_ADD_TEST(colWiseRead_12
                  MACRO colWiseRead_12.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_1-fixture)

ROOTTEST_ADD_TEST(colWiseWrite_2
                  MACRO colWiseWrite_2.C+
                  FIXTURES_SETUP root-io-stdarray-colWiseWrite_2-fixture)

ROOTTEST_ADD_TEST(colWiseRead_21
                  MACRO colWiseRead_21.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_2-fixture)

ROOTTEST_ADD_TEST(colWiseRead_22
                  MACRO colWiseRead_22.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_2-fixture)

ROOTTEST_ADD_TEST(colWiseWrite_3
                  MACRO colWiseWrite_3.C+
                  FIXTURES_SETUP root-io-stdarray-colWiseWrite_3-fixture)

ROOTTEST_ADD_TEST(colWiseRead_31
                  MACRO colWiseRead_31.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_3-fixture)

ROOTTEST_ADD_TEST(colWiseRead_32
                  MACRO colWiseRead_32.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_3-fixture)

ROOTTEST_ADD_TEST(colWiseWrite_4
                  MACRO colWiseWrite_4.C+
                  FIXTURES_SETUP root-io-stdarray-colWiseWrite_4-fixture)

ROOTTEST_ADD_TEST(colWiseRead_41
                  MACRO colWiseRead_41.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_4-fixture)

ROOTTEST_ADD_TEST(colWiseRead_42
                  MACRO colWiseRead_42.C+
                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_4-fixture)

#ROOTTEST_ADD_TEST(colWiseWrite_5
#                  MACRO colWiseWrite_5.C+
#                  FIXTURES_SETUP root-io-stdarray-colWiseWrite_5-fixture)

#ROOTTEST_ADD_TEST(colWiseRead_51
#                  MACRO colWiseRead_51.C+
#                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_5-fixture)

#ROOTTEST_ADD_TEST(colWiseRead_52
#                  MACRO colWiseRead_52.C+
#                  FIXTURES_REQUIRED root-io-stdarray-colWiseWrite_5-fixture)

ROOTTEST_ADD_TEST(checkRealData
                  MACRO checkRealData.C
                  OUTREF checkRealData.ref)

ROOTTEST_ADD_TEST(namesAnalysis
                  MACRO namesAnalysis.C
                  OUTREF namesAnalysis.ref)

ROOTTEST_ADD_TEST(aclicModelWrite
                  MACRO  aclicModelWrite.C+
                  FIXTURES_SETUP root-io-stdarray-aclicModelWrite-fixture)

ROOTTEST_ADD_TEST(modelReadDict2TFile
                  MACRO  modelReadDict2.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict2.ref
                  FIXTURES_REQUIRED root-io-stdarray-aclicModelWrite-fixture)

# ROOTTEST_ADD_TEST(modelReadDict2TXMLFile
#                   MACRO  modelReadDict2.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadDict2.ref
#                   FIXTURES_REQUIRED root-io-stdarray-aclicModelWrite-fixture)

ROOTTEST_COMPILE_MACRO(modelReadDict.C
                       FIXTURES_SETUP root-io-stdarray-modelReadDict-fixture)

ROOTTEST_ADD_TEST(modelReadDictTFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict.ref
                  FIXTURES_REQUIRED root-io-stdarray-modelReadDict-fixture)

ROOTTEST_ADD_TEST(modelReadDictTXMLFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.xml\""
                  OUTREF modelReadDict.ref
                  FIXTURES_REQUIRED root-io-stdarray-modelReadDict-fixture
                                    root-io-stdarray-aclicModelWrite-fixture)

ROOTTEST_ADD_TEST(modelReadNoDictTFile
                  MACRO  modelReadNoDict.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadNoDict.ref
                  FIXTURES_REQUIRED root-io-stdarray-aclicModelWrite-fixture)

ROOTTEST_ADD_TEST(modelCheckValues
                  MACRO modelCheckValues.C
                  FIXTURES_REQUIRED root-io-stdarray-aclicModelWrite-fixture)

# ROOTTEST_ADD_TEST(modelReadNoDictTXMLFile
#                   MACRO  modelReadNoDict.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadNoDict.ref
#                   FIXTURES_REQUIRED root-io-stdarray-aclicModelWrite-fixture)

ROOTTEST_ADD_TEST(aclic03
                  MACRO  aclic03.C+
                  OUTREF aclic03.ref)

ROOTTEST_ADD_TEST(aclic01
                  MACRO  aclic01.C+
                  OUTREF aclic01.ref)

ROOTTEST_ADD_TEST(aclic00
                  MACRO  aclic00.C+
                  OUTREF aclic00.ref)
