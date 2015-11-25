#---Copy from source to binary some of the files
configure_file(HistArray.root . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(EXCLUDE execMissing runTExMap)

ROOTTEST_ADD_TEST(execMissing
                  MACRO execMissing.C
                  OUTREF execMissing.oref
                  ERRREF execMissing.eref )

ROOTTEST_ADD_TEST(runTExMap
                  MACRO runTExMap.C
                  OUTREF runTExMap.oref
                  ERRREF runTExMap.eref )
