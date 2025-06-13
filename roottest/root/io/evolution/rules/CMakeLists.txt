ROOTTEST_ADD_TEST(AddRemoveStreamerInfo
                  MACRO execAddRemoveStreamerInfo.C
                  OUTREF execAddRemoveStreamerInfo.ref
                  LABELS longtest io)

ROOTTEST_COMPILE_MACRO(execCheckRuleRegistration.cxx
                       FIXTURES_SETUP root-io-evolution-rules-CheckRuleRegistration-fixture)

ROOTTEST_ADD_TEST(CheckRuleRegistration
                  MACRO execCheckRuleRegistration.cxx+
                  OUTREF execCheckRuleRegistration.ref
                  FIXTURES_REQUIRED root-io-evolution-rules-CheckRuleRegistration-fixture
                  LABELS longtest io)

ROOTTEST_COMPILE_MACRO(execSourceTypes.cxx
                       FIXTURES_SETUP root-io-evolution-rules-SourceTypes-fixture)

ROOTTEST_ADD_TEST(SourceTypes
                  MACRO execSourceTypes.cxx+
                  OUTREF execSourceTypes.ref
                  FIXTURES_REQUIRED root-io-evolution-rules-SourceTypes-fixture
                  LABELS longtest io)

