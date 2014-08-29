#---Copy from source to binary some of the files
configure_file(HistArray.root . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(EXCLUDE execMissing)

ROOTTEST_ADD_TEST(roottest-root-collection-execMissing
                  MACRO execMissing.C
                  OUTREF execMissing.oref
                  ERRREF execMissing.eref )
