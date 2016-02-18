ROOTTEST_ADD_TEST(rundefaultargs-interpreted
                  MACRO rundefaultargs.C
                  OUTREF defaultargs_interpreted.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(rundefaultargs-compiled
                  MACRO rundefaultargs.C+
                  OUTREF defaultargs_compiled.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runtemplateAutodict
                  MACRO runtemplateAutodict.C
                  OUTREF templateAutodict.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runoperators
                  MACRO runoperators.C
                  OUTREF operators.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runalgebra
                  MACRO runalgebra.C+
                  OUTREF algebra.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(exectemplatetemplateTest
                  MACRO exectemplatetemplateTest.cxx+
                  OUTREF exectemplatetemplateTest.ref
                  LABELS roottest regression cling)


# ROOT-7879
ROOTTEST_ADD_TEST(skipLinkDefDict
                  COMMAND ${ROOT_rootcling_CMD} -f libSkippingDict.cxx -rmf libSkipping.rootmap -rml libSkipping.so  ${CMAKE_CURRENT_SOURCE_DIR}/skippingHeader.h ${CMAKE_CURRENT_SOURCE_DIR}/skipLinkDef.h
                  OUTREF skipLinkDefDict.ref)


ROOTTEST_ADD_TEST(skipLinkDef
                  COMMAND grep clingAutoload libSkippingDict.cxx
                  DEPENDS skipLinkDefDict
                  OUTREF skippingDict.ref)



ROOTTEST_ADD_TESTDIRS()
