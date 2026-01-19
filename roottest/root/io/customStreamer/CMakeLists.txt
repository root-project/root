ROOTTEST_ADD_TEST(run
                  MACRO run.C
                  OUTREF customStreamerTest.ref)

ROOTTEST_GENERATE_DICTIONARY(StreamerThrowClass
                             StreamerThrowClass.h
                             LINKDEF StreamerThrowClass_LinkDef.h
                             FIXTURES_SETUP root-io-customStreamer-StreamerThrowClass-fixture)

# ACLiC compiled to work around problems with exceptions in Apple Silicon.
ROOTTEST_ADD_TEST(StreamerThrow
                  MACRO StreamerThrow.C+
                  OUTREF StreamerThrow.ref
                  FIXTURES_REQUIRED root-io-customStreamer-StreamerThrowClass-fixture)

ROOTTEST_GENERATE_DICTIONARY(MemberStreamerLib
                             MemberStreamerLib.h
                             LINKDEF MemberStreamerLib_LinkDef.h
                             FIXTURES_SETUP root-io-customStreamer-MemberStreamerLib-fixture)

ROOTTEST_ADD_TEST(MemberStreamer
                  MACRO testMemberStreamer.C
                  PASSRC 0
                  FIXTURES_REQUIRED root-io-customStreamer-MemberStreamerLib-fixture)

ROOTTEST_ADD_TEST(MemberStreamer2
                  MACRO testMemberStreamer2.C
                  PASSRC 0
                  FIXTURES_REQUIRED root-io-customStreamer-MemberStreamerLib-fixture)
