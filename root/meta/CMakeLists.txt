ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES Hist)

ROOTTEST_ADD_TEST(roottest-root-meta-loadernotapp
                  EXEC loadernotapp
                  OUTREF loadernotapp.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_COMPILE_MACRO(fornamespace.C)

#---Copy from source to binary some of the files
configure_file(templateAutoload.rootmap . COPYONLY)
configure_file(typelist.v5.txt . COPYONLY)
configure_file(typelist.v6.txt . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS TmpltInt0.C TmpltInt1.C TmpltFloat.C
                                TmpltNoSpec.C Event.cxx ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TESTDIRS()
