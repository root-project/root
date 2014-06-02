ROOTTEST_GENERATE_REFLEX_DICTIONARY(FullheaderParsingOnDemand FullheaderParsingOnDemand.h SELECTION FullheaderParsingOnDemand_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-autoloading-headerParsingOnDemand-runFullheaderParsingOnDemand
                  MACRO runFullheaderParsingOnDemand.C
                  OUTREF headerParsingOnDemand.ref
                  OUTCNV FullheaderParsingOnDemand_convert.sh
                  DEPENDS ${GENERATE_REFLEX_TEST})

