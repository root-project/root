ROOTTEST_ADD_TEST(execBrowseNonAccessibleType
                  OUTREF execBrowseNonAccessibleType.ref
                  MACRO execBrowseNonAccessibleType.C)

ROOTTEST_ADD_TEST(RunMyClass
                  MACRO RunMyClass.C+)

ROOTTEST_ADD_TEST(runclassref
                  MACRO runclassref.C
                  OUTREF classref.ref
                  DEPENDS RunMyClass)

ROOTTEST_ADD_TEST(runtempParse
                  MACRO runtempParse.C
                  OUTREF tempParse.ref
                  DEPENDS tempParse.C)

ROOTTEST_ADD_TEST(runpairs
                  COPY_TO_BUILDDIR  pairs_v5.root
                  MACRO runpairs.C+
                  OUTREF pairs.ref)

ROOTTEST_ADD_TEST(execInterpClassNew
                  MACRO execInterpClassNew.C
                  OUTREF execInterpClassNew.ref
                  DEPENDS classlib.cxx)

ROOTTEST_ADD_TEST(execProperties
                  MACRO execProperties.C
                  OUTREF execProperties.ref)

ROOTTEST_ADD_TEST(execState
                  COPY_TO_BUILDDIR tc_state.root libState.cxx
                  MACRO execState.C
                  OUTREF execState.ref
                  DEPENDS libState.cxx)

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
                  OUTREF execTmpltD.ref)

ROOTTEST_ADD_TEST(execTmpltD32
                  MACRO execTmpltD32.cxx+
                  OUTREF execTmpltD32.ref)

ROOTTEST_COMPILE_MACRO(execReadTmplt.cxx)

ROOTTEST_ADD_TEST(execReadTmplt
                  COPY_TO_BUILDDIR tmplt.h
                  MACRO execReadTmplt.cxx+
                  OUTREF execReadTmplt.ref
                  DEPENDS ${COMPILE_MACRO_TEST} execTmpltD32 execTmpltD)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(SimpleVector simpleVector.h SELECTION simpleVector_selection.xml)

ROOTTEST_ADD_TEST(execSimpleVector
                  MACRO execSimpleVector.C
                  OUTREF execSimpleVector.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(execDuplicate
                  MACRO execDuplicate.C
                  OUTREF execDuplicate.ref
                  DEPENDS libDuplicate.cxx)

ROOTTEST_ADD_TEST(execReadSimplePair
                  COPY_TO_BUILDDIR simplepair.root
                  MACRO execReadSimplePair.C
                  OUTREF execReadSimplePair.ref)

ROOTTEST_GENERATE_DICTIONARY(libdatamemberload
                             datamemberload.h
                             LINKDEF datamemberloadLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_ADD_TEST(datamemberload
                  MACRO datamemberload.C
                  COPY_TO_BUILDDIR datamemberload.h
                  DEPENDS libdatamemberload-build)

ROOTTEST_GENERATE_DICTIONARY(libInitOrderDups
                             InitOrderDups.h
                             LINKDEF InitOrderDupsLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP InitOrder)

ROOTTEST_GENERATE_DICTIONARY(libInitOrder
                             InitOrder.h
                             LINKDEF InitOrderLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP InitOrder)

ROOTTEST_ADD_TEST(execInitOrder
                  MACRO execInitOrder.cxx+
                  COPY_TO_BUILDDIR InitOrder.h
                  OUTREF execInitOrder.ref
                  FIXTURES_REQUIRED InitOrder)

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
