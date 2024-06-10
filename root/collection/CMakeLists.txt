#---Copy from source to binary some of the files
configure_file(HistArray.root . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(EXCLUDE execMissing runTExMap runDeleteWarning)

ROOTTEST_ADD_TEST(execMissing
                  MACRO execMissing.C
                  OUTREF execMissing.oref
                  ERRREF execMissing.eref )

if (32BIT OR MSVC)
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

ROOTTEST_ADD_TEST(runDeleteWarning
                  MACRO runDeleteWarning.C
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:0xRemoved:g
                  OUTREF DeleteWarning.ref)
