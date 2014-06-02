add_definitions(
-DClingWorkAroundTClassUpdateDouble32
)


ROOTTEST_ADD_TEST(roottest-root-meta-tclass-RunMyClass
                  MACRO RunMyClass.C+) 

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runclassref
                  MACRO runclassref.C
                  OUTREF classref.ref)

ROOTTEST_COMPILE_MACRO(tempParse.C)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runtempParse
                  MACRO runtempParse.C
                  OUTREF tempParse.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-runpairs
                  MACRO runpairs.C+
                  OUTREF pairs.ref)

ROOTTEST_COMPILE_MACRO(classlib.cxx)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execInterpClassNew
                  MACRO execInterpClassNew.C
                  OUTREF execInterpClassNew.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execProperties
                  MACRO execProperties.C
                  OUTREF execProperties.ref)

ROOTTEST_COMPILE_MACRO(libState.cxx)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execState
                  MACRO execState.C
                  OUTREF execState.ref
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execSharedPtr
                  MACRO execSharedPtr.C
                  OUTREF execSharedPtr.ref)


ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execTmpltD
                  MACRO execTmpltD.cxx+
                  OUTREF execTmpltD.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execTmpltD32
                  MACRO execTmpltD32.cxx+
                  OUTREF execTmpltD32.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-tclass-execReadTmplt
                  MACRO execReadTmplt.cxx+
                  OUTREF execReadTmplt.ref)
