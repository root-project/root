ROOTTEST_ADD_TEST(run
                  MACRO run.C
                  OUTREF customStreamerTest.ref)

ROOTTEST_GENERATE_DICTIONARY(StreamerThrowClass StreamerThrowClass.h
                             LINKDEF StreamerThrowClass_LinkDef.h
                             FIXTURES_SETUP root-io-customStreamer-StreamerThrow)
# ACLiC compiled to work around problems with exceptions in Apple Silicon.
ROOTTEST_ADD_TEST(StreamerThrow
                  MACRO StreamerThrow.C+
                  OUTREF StreamerThrow.ref
                  FIXTURES_REQUIRED root-io-customStreamer-StreamerThrow)
