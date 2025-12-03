ROOTTEST_ADD_TEST(execCheckNaming
                  MACRO execCheckNaming.C
                  OUTREF execCheckNaming${ref_suffix})

ROOTTEST_ADD_TEST(ClassEditNormalize
                  MACRO execClassEditNormalize.C
                  OUTREF execClassEditNormalize.ref)

ROOTTEST_COMPILE_MACRO(namingMatches.cxx
                       FIXTURES_SETUP root-meta-naming-namingMatches-fixture)

ROOTTEST_ADD_TEST(NamingMatches
                  MACRO execNamingMatches.cxx+
                  OUTREF execNamingMatches.ref
                  FIXTURES_REQUIRED root-meta-naming-namingMatches-fixture)

ROOTTEST_ADD_TEST(NamingMatchesNoHeader
                  MACRO execNamingMatchesNoHeader.cxx+
                  OUTREF execNamingMatchesNoHeader.ref
                  FIXTURES_REQUIRED root-meta-naming-namingMatches-fixture)

ROOTTEST_ADD_TEST(ParamPack
                  MACRO execParamPack.C
                  OUTREF execParamPack.ref)

ROOTTEST_ADD_TEST(ResolveTypedef
                  MACRO execResolveTypedef.cxx
                  OUTREF execResolveTypedef.ref)

ROOTTEST_ADD_TESTDIRS()
