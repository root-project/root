ROOTTEST_ADD_TEST(Unload
                  MACRO assertUnload.C)

ROOTTEST_ADD_TEST(ResolveAt
                  MACRO execResolveAt.C
                  OUTREF execResolveAt.ref)

ROOTTEST_ADD_TEST(memberFunc
                  MACRO runmemberFunc.C
                  OUTREF memberFunc.ref)

ROOTTEST_ADD_TEST(simpleFunc
                  MACRO runsimpleFunc.C
                  OUTREF simpleFunc.ref)
