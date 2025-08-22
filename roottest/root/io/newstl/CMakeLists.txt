ROOTTEST_COMPILE_MACRO(TestHelpers.C
                       FIXTURES_SETUP root-io-newstl-TestHelpers-fixture)

set(_holders rvecHolder vectorHolder dequeHolder listHolder setHolder multisetHolder mapHolder multimapHolder)
set(_tests rtest vtest dtest ltest stest ttest mtest ntest)

foreach(index RANGE 7)
  list(GET _holders ${index} holder)
  list(GET _tests ${index} test)
  ROOTTEST_COMPILE_MACRO(${holder}.C
                         BUILDLIB TestHelpers_C
                         FIXTURES_REQUIRED root-io-newstl-TestHelpers-fixture
                         FIXTURES_SETUP root-io-newstl-${holder}-fixture)
  ROOTTEST_COMPILE_MACRO(${test}.C
                         BUILDLIB ${holder}_C
                         FIXTURES_REQUIRED root-io-newstl-${holder}-fixture
                         FIXTURES_SETUP root-io-newstl-test-fixture)
endforeach()

if(APPLE OR MSVC)
  set(_srun_ref stlIoTestMac.ref)
else()
  set(_srun_ref stlIoTest.ref)
endif()

ROOTTEST_ADD_TEST(run
                  MACRO srun.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}/\""
                  OUTREF ${_srun_ref}
                  FIXTURES_REQUIRED root-io-newstl-test-fixture
                  FIXTURES_SETUP root-io-newstl-run-fixture)

# failing tests, crashes ROOT, was disabled in Makefile
ROOTTEST_ADD_TEST(nolib
                  MACRO readNoLib.C
                  MACROARG "\"vector.root\""
                  OUTREF stlNoLibTest.ref
                  WILLFAIL
                  FIXTURES_REQUIRED root-io-newstl-run-fixture)

ROOTTEST_COMPILE_MACRO(ComplexTest.h
                       FIXTURES_SETUP root-io-newstl-ComplexTest_h-fixture)

ROOTTEST_COMPILE_MACRO(sample_bx_classes.C
                       FIXTURES_SETUP root-io-newstl-sample_bx_classes-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_complextest_load -e "(void)gROOT->ProcessLine(\".L ComplexTest.h+\")")
  set(_sample_bx_classes_load -e "(void)gROOT->ProcessLine(\".L sample_bx_classes.C+\")")
endif()

ROOTTEST_ADD_TEST(ComplexTest
                  MACRO runComplexTest.C
                  ROOTEXE_OPTS ${_complextest_load}
                  OUTREF ComplexTest.ref
                  FIXTURES_REQUIRED root-io-newstl-ComplexTest_h-fixture)

ROOTTEST_ADD_TEST(SampleFile
                  MACRO sample_writer.C
                  ROOTEXE_OPTS ${_sample_bx_classes_load}
                  FIXTURES_REQUIRED root-io-newstl-sample_bx_classes-fixture
                  FIXTURES_SETUP root-io-newstl-SampleFile-fixture)

ROOTTEST_ADD_TEST(emptyCollection
                  MACRO runemptyCollection.C
                  OUTREF emptyCollection.ref
                  FIXTURES_REQUIRED root-io-newstl-sample_bx_classes-fixture
                                    root-io-newstl-SampleFile-fixture)

ROOTTEST_ADD_TEST(emulatedWrite
                  COMMAND ${ROOT_hadd_CMD} -f result.root cmsJets_0.root cmsJets_1.root
                  COPY_TO_BUILDDIR cmsJets_0.root cmsJets_1.root
                  OUTREF emulatedWrite.ref)

ROOTTEST_ADD_TEST(base
                  MACRO runbase.C+
                  OUTREF base.ref)

ROOTTEST_ADD_TEST(writetest_emulvector
                  MACRO writetest_emulvector.C
                  OUTREF emulvector.root.ref
                  FIXTURES_SETUP root-io-newstl-writetest_emulvector-fixture)

ROOTTEST_ADD_TEST(readtest_emulvector
                  MACRO runEmulvector.C
                  OUTREF Emulvector.ref
                  FIXTURES_REQUIRED root-io-newstl-writetest_emulvector-fixture)

ROOTTEST_ADD_TEST(writeMemberWise
                  MACRO writeMemberWise.cxx+
                  FIXTURES_SETUP root-io-newstl-writeMemberWise-fixture)

ROOTTEST_ADD_TEST(readMemberWise
                  MACRO execMemberWise.cxx+
                  OUTREF execMemberWise.ref
                  FIXTURES_REQUIRED root-io-newstl-writeMemberWise-fixture)

ROOTTEST_ADD_TEST(buildPospelov
                  MACRO buildPospelov.C
                  COPY_TO_BUILDDIR pospelov.2010.mc10_7TeV.pool.root PospelovPrintFile.C
                  FIXTURES_SETUP root-io-newstl-buildPospelov-fixture)

ROOTTEST_ADD_TEST(printPospelov
                  MACRO execPospelovPrintFile.C
                  OUTREF execPospelovPrintFile.ref
                  FIXTURES_REQUIRED root-io-newstl-buildPospelov-fixture)

