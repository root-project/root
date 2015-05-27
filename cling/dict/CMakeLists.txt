ROOTTEST_ADD_TEST(roottest-cling-dict-rundefaultargs-interpreted
                  MACRO rundefaultargs.C
                  OUTREF defaultargs_interpreted.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-dict-rundefaultargs-compiled
                  MACRO rundefaultargs.C+
                  OUTREF defaultargs_compiled.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-dict-runtemplateAutodict
                  MACRO runtemplateAutodict.C
                  OUTREF templateAutodict.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-dict-runoperators
                  MACRO runoperators.C
                  OUTREF operators.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-dict-runalgebra
                  MACRO runalgebra.C+
                  OUTREF algebra.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-dict-exectemplatetemplateTest
                  MACRO exectemplatetemplateTest.cxx+
                  OUTREF exectemplatetemplateTest.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TESTDIRS()
