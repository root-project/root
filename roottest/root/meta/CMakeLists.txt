ROOTTEST_ADD_TEST(getFuncBody
                  MACRO getFuncBody.C)

ROOTTEST_ADD_TEST(countIncludePaths
                  MACRO countIncludePaths.C)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(expressiveErrorMessages
                                    expressiveErrorMessages.h
                                    SELECTION expressiveErrorMessages_selection.xml
                                    FIXTURES_SETUP root-meta-expressiveErrorMessages-fixture)

ROOTTEST_ADD_TEST(expressiveErrorMessages
                  ENABLE_IF pch
                  MACRO  expressiveErrorMessages.C
                  OUTREF expressiveErrorMessages.ref
                  PASSRC 1
                  OUTCNV expressiveErrorMessages_filter.sh
                  FIXTURES_REQUIRED root-meta-expressiveErrorMessages-fixture)

ROOTTEST_ADD_TEST(clingTErrorDiagnostics
                  MACRO clingTErrorDiagnostics.C
                  OUTREF clingTErrorDiagnostics.ref)

ROOTTEST_ADD_TEST(ROOT5268
                  MACRO ROOT-5268.C
                  PASSREGEX "error: unknown type name 'Tbrowser'")

ROOTTEST_ADD_TEST(rlibmap
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/rlibmapLauncher.py
                  PASSRC 1
                  OUTREF  rlibmap.ref)

ROOTTEST_ADD_TEST(execpragmasTest
                  MACRO execpragmasTest.C+
                  OUTREF execpragmasTest.ref)

ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES ROOT::Core ROOT::Hist
                             FIXTURES_SETUP root-meta-loadernotapp-fixture)

ROOTTEST_ADD_TEST(loadernotapp
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/loadernotapp
                  OUTREF loadernotapp.ref
                  FIXTURES_REQUIRED root-meta-loadernotapp-fixture)

foreach(macro fornamespace ANSTmpltInt TmpltInt0 TmpltInt1 TmpltFloat TmpltNoSpec)
  set(_fixture root-meta-${macro}-fixture)
  list(APPEND _fixture_list ${_fixture})
  ROOTTEST_COMPILE_MACRO(${macro}.C FIXTURES_SETUP ${_fixture})
endforeach()

#---Copy from source to binary some of the files
configure_file(templateAutoload.rootmap . COPYONLY)
configure_file(typelist.v5.txt . COPYONLY)
configure_file(typelist.v6.txt . COPYONLY)
configure_file(typelist.v6.cxx17.txt . COPYONLY)
configure_file(typelist_win32.v5.txt . COPYONLY)
configure_file(typelist_win32.v6.txt . COPYONLY)
configure_file(typelist_win32.v6.cxx17.txt . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(EXCLUDE runMemberComments runautoload
                        FIXTURES_REQUIRED ${_fixture_list})

ROOTTEST_ADD_TEST(runMemberComments
                  MACRO runMemberComments.C
                  OUTREF MemberComments${ref_suffix})

ROOTTEST_ADD_TEST(namespaceconflict
                  MACRO execnamespaceconflict.C+)

ROOTTEST_COMPILE_MACRO(${CMAKE_SOURCE_DIR}/roottest/root/io/event/Event.cxx
                       FIXTURES_SETUP root-meta-Event-fixture)

ROOTTEST_ADD_TEST(runautoload
                  MACRO runautoload.C
                  OUTREF autoload${ref_suffix}
                  FIXTURES_REQUIRED root-meta-Event-fixture)

ROOTTEST_COMPILE_MACRO(hello_ROOT.C
                       FIXTURES_SETUP root-meta-hello_ROOT-fixture)

if(MSVC)
    set(libsuf dll)
else()
    set(libsuf so)
endif()

ROOTTEST_ADD_TEST(hello_ROOT_C
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/hello_ROOT_C.${libsuf}
                  OUTREF hello_ROOT_C_${libsuf}.ref
                  FIXTURES_REQUIRED root-meta-hello_ROOT-fixture)

ROOTTEST_ADD_TESTDIRS()
