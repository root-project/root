ROOTTEST_ADD_TEST(V5Formula
                  MACRO execV5Formula.C+
                  COPY_TO_BUILDDIR v5formula_clones.root
                  OUTREF execV5Formula.ref)

ROOTTEST_GENERATE_DICTIONARY(libRtObjDict
                             RtObj.h
                             LINKDEF RtObjLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-clones-RtObjDict-fixture)

ROOTTEST_GENERATE_EXECUTABLE(tester
                             tester.C
                             LIBRARIES Core RIO libRtObjDictlibgen
                             FIXTURES_REQUIRED root-io-clones-RtObjDict-fixture
                             FIXTURES_SETUP root-io-clones-tester-fixture)

ROOTTEST_ADD_TEST(clone
                  EXEC ./tester
                  OUTREF clone.ref
                  FIXTURES_REQUIRED root-io-clones-tester-fixture)
