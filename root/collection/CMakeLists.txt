ROOTTEST_ADD_AUTOMACROS()

#ROOTTEST_ADD_TEST(roottest-root-collection-runtbits  MACRO runtbits.C OUTREF tbits.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-runTExMap MACRO runTExMap.C+ OUTREF TExMap.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-runTMap   MACRO runTMap.C+ OUTREF TMap.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-runCATTreeClear MACRO runCATTreeClear.C OUTREF CATTreeClear.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-execMissing MACRO execMissing.C OUTREF execMissing.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-execTClonesArrayAbsorb.C MACRO execTClonesArrayAbsorb.C OUTREF execTClonesArrayAbsorb.ref)
#ROOTTEST_ADD_TEST(roottest-root-collection-execHistArray MACRO execHistArray.cxx+ OUTREF execHistArray.ref)
