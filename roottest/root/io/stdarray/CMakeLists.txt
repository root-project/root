ROOTTEST_ADD_TESTDIRS()

# More coverage for ROOT-8736
ROOTTEST_ADD_TEST(arrTemplMemberWrite
                  MACRO testArrOfTemplateInst.C+
                  OUTREF testArrOfTemplateInst.ref)

ROOTTEST_ADD_TEST(colWiseWrite_1
                  MACRO colWiseWrite_1.C+)

ROOTTEST_ADD_TEST(colWiseRead_11
                  MACRO colWiseRead_11.C+
                  DEPENDS colWiseWrite_1)

ROOTTEST_ADD_TEST(colWiseRead_12
                  MACRO colWiseRead_12.C+
                  DEPENDS colWiseWrite_1)

ROOTTEST_ADD_TEST(colWiseWrite_2
                  MACRO colWiseWrite_2.C+)

ROOTTEST_ADD_TEST(colWiseRead_21
                  MACRO colWiseRead_21.C+
                  DEPENDS colWiseWrite_2)

ROOTTEST_ADD_TEST(colWiseRead_22
                  MACRO colWiseRead_22.C+
                  DEPENDS colWiseWrite_2)

ROOTTEST_ADD_TEST(colWiseWrite_3
                  MACRO colWiseWrite_3.C+)

ROOTTEST_ADD_TEST(colWiseRead_31
                  MACRO colWiseRead_31.C+
                  DEPENDS colWiseWrite_3)

ROOTTEST_ADD_TEST(colWiseRead_32
                  MACRO colWiseRead_32.C+
                  DEPENDS colWiseWrite_3)

ROOTTEST_ADD_TEST(colWiseWrite_4
                  MACRO colWiseWrite_4.C+)

ROOTTEST_ADD_TEST(colWiseRead_41
                  MACRO colWiseRead_41.C+
                  DEPENDS colWiseWrite_4)

ROOTTEST_ADD_TEST(colWiseRead_42
                  MACRO colWiseRead_42.C+
                  DEPENDS colWiseWrite_4)

#ROOTTEST_ADD_TEST(colWiseWrite_5
#                  MACRO colWiseWrite_5.C+)

#ROOTTEST_ADD_TEST(colWiseRead_51
#                  MACRO colWiseRead_51.C+
#                  DEPENDS colWiseWrite_5)

#ROOTTEST_ADD_TEST(colWiseRead_52
#                  MACRO colWiseRead_52.C+
#                  DEPENDS colWiseWrite_5)

ROOTTEST_ADD_TEST(checkRealData
                  MACRO checkRealData.C
                  OUTREF checkRealData.ref)

ROOTTEST_ADD_TEST(namesAnalysis
                  MACRO namesAnalysis.C
                  OUTREF namesAnalysis.ref)

ROOTTEST_ADD_TEST(modelReadDict2TFile
                  MACRO  modelReadDict2.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict2.ref
                  DEPENDS aclicModelWrite)

# ROOTTEST_ADD_TEST(modelReadDict2TXMLFile
#                   MACRO  modelReadDict2.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadDict2.ref
#                   DEPENDS aclicModelWrite)

# The dependency is there not to compile the macros simultaneously
ROOTTEST_ADD_TEST(modelReadDictTFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.root\""
                  OUTREF modelReadDict.ref
                  DEPENDS modelReadDictTXMLFile)

ROOTTEST_ADD_TEST(modelReadDictTXMLFile
                  MACRO  modelReadDict.C+
                  MACROARG "\"model.xml\""
                  OUTREF modelReadDict.ref
                  DEPENDS aclicModelWrite)

ROOTTEST_ADD_TEST(modelReadNoDictTFile
                  MACRO  modelReadNoDict.C
                  MACROARG "\"model.root\""
                  OUTREF modelReadNoDict.ref
                  DEPENDS aclicModelWrite)

ROOTTEST_ADD_TEST(modelCheckValues
                  MACRO modelCheckValues.C
                  DEPENDS aclicModelWrite)

# ROOTTEST_ADD_TEST(modelReadNoDictTXMLFile
#                   MACRO  modelReadNoDict.C
#                   MACROARG "\"model.xml\""
#                   OUTREF modelReadNoDict.ref
#                   DEPENDS aclicModelWrite)

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
