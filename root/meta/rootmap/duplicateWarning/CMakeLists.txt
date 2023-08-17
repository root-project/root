configure_file(rfile1.rootmap . COPYONLY)
configure_file(rfile2.rootmap . COPYONLY)

ROOTTEST_ADD_TEST(myclassduplicated
                  COMMAND root -b -q -l
                  OUTREF  warning_duplicate_autoload_key.ref)
