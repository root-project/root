# A test for issue #9058
configure_file(execscript.py . COPYONLY)
ROOTTEST_ADD_TEST(execscript
                  MACRO  execscript.C
                  FAILREGEX "FAILURE")
