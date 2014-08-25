ROOTTEST_ADD_TEST(roottest-root-meta-tclass-RunMyClass
                  MACRO RunMyClass.C+) 

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runclassref
                  MACRO runclassref.C
                  OUTREF classref.ref
                  DEPENDS roottest-root-meta-tclass-RunMyClass)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runtempParse
                  MACRO runtempParse.C
                  OUTREF tempParse.ref
                  DEPENDS tempParse.C)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runpairs
                  COPY_TO_BUILDDIR  pairs_v5.root
                  MACRO runpairs.C+
                  OUTREF pairs.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execInterpClassNew
                  MACRO execInterpClassNew.C
                  OUTREF execInterpClassNew.ref
                  DEPENDS classlib.cxx)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execProperties
                  MACRO execProperties.C
                  OUTREF execProperties.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execState
                  COPY_TO_BUILDDIR tc_state.root libState.cxx
                  MACRO execState.C
                  OUTREF execState.ref
                  DEPENDS libState.cxx)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execSharedPtr
                  MACRO execSharedPtr.C
                  OUTREF execSharedPtr.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execTmpltD
                  MACRO execTmpltD.cxx+
                  OUTREF execTmpltD.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execTmpltD32
                  MACRO execTmpltD32.cxx+
                  OUTREF execTmpltD32.ref)

ROOTTEST_COMPILE_MACRO(execReadTmplt.cxx)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execReadTmplt
                  COPY_TO_BUILDDIR tmplt.h
                  MACRO execReadTmplt.cxx+
                  OUTREF execReadTmplt.ref
                  DEPPENDS ${COMPILE_MACRO_TEST} roottest-root-meta-tclass-execTmpltD32 roottest-root-meta-tclass-execTmpltD)

ROOTTEST_ADD_TESTDIRS()
