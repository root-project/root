ROOTTEST_GENERATE_DICTIONARY(libMapPairName
                             MapPairName.h
                             LINKDEF MapPairNameLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-mapPairName-dict)

ROOTTEST_ADD_TEST(mapPairName
                  MACRO mapPairName.C
                  FIXTURES_REQUIRED root-meta-tclass-mapPairName-dict)
