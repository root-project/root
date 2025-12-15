ROOTTEST_GENERATE_REFLEX_DICTIONARY(ioReadRuleEquivalentCode
                                    ioReadRuleEquivalentCode.h
                                    SELECTION ioReadRuleEquivalentCode_selection.xml
                                    FIXTURES_SETUP root-io-evolution-equivalent-ioReadRuleEquivalentCode-fixture)

ROOTTEST_ADD_TEST(ioReadRuleEquivalentCode
                  MACRO execioReadRuleEquivalentCode.C
                  OUTREF execioReadRuleEquivalentCode.ref
                  COPY_TO_BUILDDIR ioReadRuleEquivalentCode.root
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-equivalent-ioReadRuleEquivalentCode-fixture)
