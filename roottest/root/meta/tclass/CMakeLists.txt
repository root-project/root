ROOTTEST_ADD_TEST(execBrowseNonAccessibleType
                  OUTREF execBrowseNonAccessibleType.ref
                  MACRO execBrowseNonAccessibleType.C)

ROOTTEST_ADD_TEST(RunMyClass
                  MACRO RunMyClass.C+
                  FIXTURES_SETUP root-meta-tclass-RunMyClas-fixture)

ROOTTEST_ADD_TEST(runclassref
                  MACRO runclassref.C
                  OUTREF classref.ref
                  FIXTURES_REQUIRED root-meta-tclass-RunMyClas-fixture)

ROOTTEST_COMPILE_MACRO(tempParse.C
                       FIXTURES_SETUP root-meta-tclass-tempParse-fixture)

ROOTTEST_ADD_TEST(runtempParse
                  MACRO runtempParse.C
                  OUTREF tempParse.ref
                  FIXTURES_REQUIRED root-meta-tclass-tempParse-fixture)

ROOTTEST_ADD_TEST(runpairs
                  COPY_TO_BUILDDIR  pairs_v5.root
                  MACRO runpairs.C+
                  OUTREF pairs.ref)

ROOTTEST_COMPILE_MACRO(classlib.cxx
                       FIXTURES_SETUP root-meta-tclass-classlib-fixture)

ROOTTEST_ADD_TEST(execInterpClassNew
                  MACRO execInterpClassNew.C
                  OUTREF execInterpClassNew.ref
                  FIXTURES_REQUIRED root-meta-tclass-classlib-fixture)

ROOTTEST_ADD_TEST(execProperties
                  MACRO execProperties.C
                  OUTREF execProperties.ref)

ROOTTEST_COMPILE_MACRO(libState.cxx
                       FIXTURES_SETUP root-meta-tclass-libState-fixture)

ROOTTEST_ADD_TEST(execState
                  COPY_TO_BUILDDIR tc_state.root
                  MACRO execState.C
                  OUTREF execState.ref
                  FIXTURES_REQUIRED root-meta-tclass-libState-fixture)

ROOTTEST_ADD_TEST(execSharedPtr
                  MACRO execSharedPtr.C
                  ERRREF execSharedPtr.eref)

ROOTTEST_ADD_TEST(execMTInit
                  MACRO execMTInit.C
                  ERRREF execMTInit.eref)

ROOTTEST_ADD_TEST(execTClassRehash
                  MACRO execTClassRehash.C
                  ERRREF execTClassRehash.eref)

ROOTTEST_ADD_TEST(execTmpltD
                  MACRO execTmpltD.cxx+
                  OUTREF execTmpltD.ref
                  FIXTURES_SETUP root-meta-tclass-execTmpltD-fixture)

ROOTTEST_ADD_TEST(execTmpltD32
                  MACRO execTmpltD32.cxx+
                  OUTREF execTmpltD32.ref
                  FIXTURES_SETUP root-meta-tclass-execTmpltD32-fixture)

ROOTTEST_COMPILE_MACRO(execReadTmplt.cxx
                       FIXTURES_SETUP root-meta-tclass-execReadTmplt-fixture)

ROOTTEST_ADD_TEST(execReadTmplt
                  COPY_TO_BUILDDIR tmplt.h
                  MACRO execReadTmplt.cxx+
                  OUTREF execReadTmplt.ref
                  FIXTURES_REQUIRED root-meta-tclass-execReadTmplt-fixture
                                    root-meta-tclass-execTmpltD-fixture
                                    root-meta-tclass-execTmpltD32-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(SimpleVector simpleVector.h SELECTION simpleVector_selection.xml
                                    FIXTURES_SETUP root-meta-tclass-SimpleVector-fixture)

ROOTTEST_ADD_TEST(execSimpleVector
                  MACRO execSimpleVector.C
                  OUTREF execSimpleVector.ref
                  FIXTURES_REQUIRED root-meta-tclass-SimpleVector-fixture)

ROOTTEST_COMPILE_MACRO(libDuplicate.cxx
                       FIXTURES_SETUP root-meta-tclass-libDuplicate-fixture)

ROOTTEST_ADD_TEST(execDuplicate
                  MACRO execDuplicate.C
                  OUTREF execDuplicate.ref
                  FIXTURES_REQUIRED root-meta-tclass-libDuplicate-fixture)

ROOTTEST_ADD_TEST(execReadSimplePair
                  COPY_TO_BUILDDIR simplepair.root
                  MACRO execReadSimplePair.C
                  OUTREF execReadSimplePair.ref)

ROOTTEST_GENERATE_DICTIONARY(libdatamemberload
                             datamemberload.h
                             LINKDEF datamemberloadLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-libdatamemberload-fixture)

ROOTTEST_ADD_TEST(datamemberload
                  MACRO datamemberload.C
                  COPY_TO_BUILDDIR datamemberload.h
                  FIXTURES_REQUIRED root-meta-tclass-libdatamemberload-fixture)

ROOTTEST_GENERATE_DICTIONARY(libInitOrderDups
                             InitOrderDups.h
                             LINKDEF InitOrderDupsLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-libInitOrderDups-fixture)

ROOTTEST_GENERATE_DICTIONARY(libInitOrder
                             InitOrder.h
                             LINKDEF InitOrderLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-libInitOrder-fixture)

ROOTTEST_ADD_TEST(execInitOrder
                  MACRO execInitOrder.cxx+
                  COPY_TO_BUILDDIR InitOrder.h
                  OUTREF execInitOrder.ref
                  FIXTURES_REQUIRED root-meta-tclass-libInitOrderDups-fixture root-meta-tclass-libInitOrder-fixture)

if(NOT coverage)
  ROOTTEST_ADD_TEST(tclassStl
                    MACRO tclassStl.cxx+
                    OUTREF tclassStl.ref
                    RUN_SERIAL)
endif()

ROOTTEST_ADD_TEST(TClassAtTearDown
                  MACRO execTClassAtTearDown.C
		  COPY_TO_BUILDDIR empty.root
		  OUTREF execTClassAtTearDown.ref)

ROOTTEST_GENERATE_EXECUTABLE(TFileTearDown TFileTearDown.cxx
                             LIBRARIES Core RIO Hist Rint
                             FIXTURES_SETUP TFileTearDownExec)

ROOTTEST_ADD_TEST(TFileTearDown
                  EXEC ./TFileTearDown
                  FIXTURES_REQUIRED TFileTearDownExec)

ROOTTEST_ADD_TESTDIRS()
