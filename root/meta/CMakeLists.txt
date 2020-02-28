ROOTTEST_ADD_TEST(getFuncBody
                  MACRO getFuncBody.C)

ROOTTEST_ADD_TEST(countIncludePaths
                  MACRO countIncludePaths.C)
#ROOT-6752
# This test checks that the payload is properly printed in case of errors for
# debugging reasons. With modules, the concept of payload is not really valid and
# the test has no reason to run.
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_GENERATE_REFLEX_DICTIONARY(expressiveErrorMessages expressiveErrorMessages.h SELECTION expressiveErrorMessages_selection.xml)

ROOTTEST_ADD_TEST(expressiveErrorMessages
                  ENABLE_IF pch
                  MACRO  expressiveErrorMessages.C
                  OUTREF expressiveErrorMessages.ref
                  PASSRC 1
                  OUTCNV expressiveErrorMessages_filter.sh
                  DEPENDS ${GENERATE_REFLEX_TEST})
endif()

ROOTTEST_ADD_TEST(rlibmap
                  EXEC rlibmapLauncher.py
                  PASSRC 1
                  OUTREF  rlibmap.ref)

ROOTTEST_ADD_TEST(execpragmasTest
                  MACRO execpragmasTest.C+
                  OUTREF execpragmasTest.ref)

ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES Core Hist)

ROOTTEST_ADD_TEST(loadernotapp
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/loadernotapp
                  OUTREF loadernotapp.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_COMPILE_MACRO(fornamespace.C)

#---Copy from source to binary some of the files
configure_file(templateAutoload.rootmap . COPYONLY)
configure_file(typelist.v5.txt . COPYONLY)
configure_file(typelist.v6.txt . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ANSTmpltInt.C TmpltInt0.C TmpltInt1.C TmpltFloat.C
                                TmpltNoSpec.C Event.cxx ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(drawing
                  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/DrawTest.sh
                  OUTCNVCMD grep -v -e "--- Switching to" -e "--- Using "
                                    -e "^make" -e "Event.*root made" -e "dt.*made" -e ExecutionContext
                  OUTREF drawing.ref
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})

ROOTTEST_ADD_TEST(namespaceconflict
                  MACRO execnamespaceconflict.C+)


ROOTTEST_ADD_TESTDIRS()
