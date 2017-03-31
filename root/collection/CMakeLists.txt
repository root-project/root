#---Copy from source to binary some of the files
configure_file(HistArray.root . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(EXCLUDE execMissing runTExMap)

ROOTTEST_ADD_TEST(TSeqIterators
                  MACRO TSeqIterators.C)

ROOTTEST_ADD_TEST(arrayViewBasic
                  MACRO arrayViewBasic.C
                  OUTREF arrayViewBasic.ref)

ROOTTEST_ADD_TEST(execMissing
                  MACRO execMissing.C
                  OUTREF execMissing.oref
                  ERRREF execMissing.eref )

if (32BIT)
ROOTTEST_ADD_TEST(runTExMap
                  MACRO runTExMap.C
                  OUTREF runTExMap32.oref
                  ERRREF runTExMap32.eref )
else()
ROOTTEST_ADD_TEST(runTExMap
                  MACRO runTExMap.C
                  OUTREF runTExMap.oref
                  ERRREF runTExMap.eref )
endif()
