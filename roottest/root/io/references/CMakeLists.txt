ROOTTEST_ADD_TESTDIRS()


ROOTTEST_ADD_TEST(assertManyRefs
                  MACRO assertManyRefs.C
                  OUTREF assertManyRefs.ref)

ROOTTEST_COMPILE_MACRO(A.C
                       FIXTURES_SETUP root-io-references-A-fixture)

ROOTTEST_COMPILE_MACRO(lotsRef.C
                       BUILDLIB A_C
                       FIXTURES_REQUIRED root-io-references-A-fixture
                       FIXTURES_SETUP root-io-references-lotsRef-fixture)

ROOTTEST_ADD_TEST(lotsRef300
                  MACRO lotsRef.C+
                  MACROARG 300
                  FIXTURES_REQUIRED root-io-references-lotsRef-fixture
                                    root-io-references-A-fixture
                  FIXTURES_SETUP root-io-references-lotsRef300-fixture)

ROOTTEST_ADD_TEST(lotsRef2
                  MACRO lotsRef.C+
                  MACROARG 2
                  FIXTURES_REQUIRED root-io-references-lotsRef300-fixture
                  FIXTURES_SETUP root-io-references-lotsRef2-fixture)

ROOTTEST_ADD_TEST(lotsRef1
                  MACRO lotsRef.C+
                  MACROARG 1
                  OUTREF lotsRef1.ref
                  FIXTURES_REQUIRED root-io-references-lotsRef2-fixture)

ROOTTEST_ADD_TEST(lotsRef
                  MACRO runlotsRef.C
                  OUTREF lotsRef.ref
                  FIXTURES_REQUIRED root-io-references-lotsRef2-fixture)

ROOTTEST_COMPILE_MACRO(refarray.C
                       FIXTURES_SETUP root-io-references-refarray-fixture)

ROOTTEST_ADD_TEST(refarray-create
                  MACRO refarray.C+
                  MACROARG 1
                  FIXTURES_REQUIRED root-io-references-refarray-fixture
                  FIXTURES_SETUP root-io-references-refarray-create-fixture)

ROOTTEST_ADD_TEST(refarray
                  MACRO refarray.C+
                  OUTREF refarray.ref
                  OUTCNVCMD sed -e s,at[[:space:]]0x[0-9a-fA-F]*[[:space:]]is,at_na_is,g -e s,ProcessID.*,na,g
                  FIXTURES_REQUIRED root-io-references-refarray-create-fixture)

ROOTTEST_COMPILE_MACRO(RefArrayCompress.cc
                       FIXTURES_SETUP root-io-references-RefArrayCompress-fixture)

ROOTTEST_ADD_TEST(execWriteRefArrayCompress
                  MACRO execWriteRefArrayCompress.cxx+
                  ROOTEXE_OPTS -e ".L RefArrayCompress.cc+"
                  OUTREF execWriteRefArrayCompress.ref
                  FIXTURES_REQUIRED root-io-references-RefArrayCompress-fixture)

# dictionary generation does not work for provided class,
# in Makefile was placed in failing branch
if(broken_tests)
  ROOTTEST_GENERATE_DICTIONARY(libRoottestIoReferencesMaster
                                MyClass.hh
                                LINKDEF MasterLinkDef.hh
                                NO_CXXMODULE)
endif()
