#ROOT_BUILD_COMPILE_DICT(myclbad BUILDOBJ badDict.C FILES bad.h mycl.C LinkDef.h)
#ROOT_BUILD_COMPILE_DICT(myclgood BUILDOBJ goodDict.C FILES mycl.C LinkDef.h)
#ROOT_ADD_ROOTTEST(cling-typedef_global-scopeTest MACRO run.C OUTREF output.ref-cling OUTCNV scopeTest_conversion.sh)
#ROOT_ADD_ROOTTEST(cling-typedef_global-scopeTest2 MACRO rungood.C OUTREF output.ref-cling OUTCNVCMD grep -v Processing)
