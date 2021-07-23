ROOTTEST_ADD_TEST(getFuncBody
                  MACRO getFuncBody.C)

ROOTTEST_ADD_TEST(countIncludePaths
                  MACRO countIncludePaths.C)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(expressiveErrorMessages expressiveErrorMessages.h SELECTION expressiveErrorMessages_selection.xml)

ROOTTEST_ADD_TEST(expressiveErrorMessages
                  ENABLE_IF pch
                  MACRO  expressiveErrorMessages.C
                  OUTREF expressiveErrorMessages.ref
                  PASSRC 1
                  OUTCNV expressiveErrorMessages_filter.sh
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(clingTErrorDiagnostics
                  MACRO clingTErrorDiagnostics.C
                  OUTREF clingTErrorDiagnostics.ref)

ROOTTEST_ADD_TEST(rlibmap
                  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/rlibmapLauncher.py
                  PASSRC 1
                  OUTREF  rlibmap.ref)

ROOTTEST_ADD_TEST(execpragmasTest
                  MACRO execpragmasTest.C+
                  OUTREF execpragmasTest.ref)

ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES ROOT::Core ROOT::Hist)

ROOTTEST_ADD_TEST(loadernotapp
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/loadernotapp
                  OUTREF loadernotapp.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_COMPILE_MACRO(fornamespace.C)

#---Copy from source to binary some of the files
configure_file(templateAutoload.rootmap . COPYONLY)
configure_file(typelist.v5.txt . COPYONLY)
configure_file(typelist.v6.txt . COPYONLY)
configure_file(typelist_win32.v5.txt . COPYONLY)
configure_file(typelist_win32.v6.txt . COPYONLY)
configure_file(typelist_win32.v6.cxx17.txt . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ANSTmpltInt.C TmpltInt0.C TmpltInt1.C TmpltFloat.C
                                TmpltNoSpec.C Event.cxx ${COMPILE_MACRO_TEST}
                                EXCLUDE runMemberComments runautoload)
ROOTTEST_ADD_TEST(runMemberComments
                  MACRO runMemberComments.C
                  OUTREF MemberComments${ref_suffix})

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(drawing
                      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/DrawTest.sh
                      OUTCNVCMD grep -v -e "--- Switching to" -e "--- Using "
                                        -e "^make" -e "Event.*root made" -e "dt.*made" -e ExecutionContext
                      OUTREF drawing.ref
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ROOTTEST_ADD_TEST(namespaceconflict
                  MACRO execnamespaceconflict.C+)

ROOTTEST_ADD_TEST(runautoload
                  MACRO runautoload.C
                  DEPENDS ${COMPILE_MACRO_TEST}
                  OUTREF autoload${ref_suffix})

ROOTTEST_ADD_TESTDIRS()
