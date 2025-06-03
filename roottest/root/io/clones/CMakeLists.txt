ROOTTEST_ADD_TEST(V5Formula
  MACRO execV5Formula.C+
  COPY_TO_BUILDDIR v5formula_clones.root
  OUTREF execV5Formula.ref)

if(broken_tests)
  ROOTTEST_GENERATE_DICTIONARY(libRtObjDict
    RtObj.h
    LINKDEF RtObjLinkDef.h
    NO_ROOTMAP NO_CXXMODULE
    FIXTURES_SETUP RtObjDict_fixture)

  ROOTTEST_GENERATE_EXECUTABLE(tester
    tester.C
    LIBRARIES Core libRtObjDictlibgen
    FIXTURES_REQUIRED RtObjDict_fixture
    FIXTURES_SETUP RtObj_tester)

  ROOTTEST_ADD_TEST(clone
    WILLFAIL
    EXEC ./tester
    COPY_TO_BUILDDIR RtObj.root
    FIXTURES_REQUIRED RtObj_tester)
endif()