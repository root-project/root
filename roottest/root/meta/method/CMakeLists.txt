ROOTTEST_ADD_TEST(exactMatch
                  MACRO runexactMatch.C
                  OUTREF exactMatch.ref)

ROOTTEST_ADD_TEST(ConstMethod
                  MACRO execConstMethod.cxx+
                  OUTREF execConstMethod.ref)

ROOTTEST_ADD_TEST(Constructor
                  MACRO execConstructor.cxx+
                  OUTREF execConstructor.ref)

ROOTTEST_ADD_TEST(CreateString
                  MACRO execCreateString.C
                  OUTREF execCreateString.ref)

ROOTTEST_ADD_TEST(Inlined
                  MACRO execInlined.cxx+
                  OUTREF execInlined.ref)

ROOTTEST_ADD_TEST(ReuseMethod
                  MACRO execReuseMethod.cxx
                  OUTREF execReuseMethod.ref)

ROOTTEST_ADD_TEST(Template
                  MACRO execTemplate.C
                  OUTREF execTemplate.ref)

ROOTTEST_ADD_TESTDIRS()
