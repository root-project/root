ROOTTEST_ADD_TEST(roottest-root-meta-rlibmap
                  COMMAND rlibmapLauncher.py
                  PASSRC 1
                  OUTREF  rlibmap.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-execpragmasTest
                  MACRO execpragmasTest.C+
                  OUTREF execpragmasTest.ref)

ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES Core Hist)

ROOTTEST_ADD_TEST(roottest-root-meta-loadernotapp
                  EXEC loadernotapp
                  OUTREF loadernotapp.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_COMPILE_MACRO(fornamespace.C)

#---Copy from source to binary some of the files
configure_file(templateAutoload.rootmap . COPYONLY)
configure_file(typelist.v5.txt . COPYONLY)
configure_file(typelist.v6.txt . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ANSTmpltInt.C TmpltInt0.C TmpltInt1.C TmpltFloat.C
                                TmpltNoSpec.C Event.cxx ${COMPILE_MACRO_TEST})


ROOTTEST_ADD_TEST(roottest-root-meta-drawing
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/DrawTest.sh
                  OUTCNVCMD grep -v -e "--- Switching to" -e "--- Using "
                                    -e "^make" -e "Event.*root made" -e "dt.*made" -e ExecutionContext
                  OUTREF drawing.ref
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})

ROOTTEST_ADD_TEST(roottest-root-meta-namespaceconflict
                  MACRO execnamespaceconflict.C+)


ROOTTEST_ADD_TESTDIRS()
