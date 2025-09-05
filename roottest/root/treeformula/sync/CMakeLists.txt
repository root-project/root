ROOTTEST_ADD_TEST(alias
                  MACRO runalias.C+
                  OUTREF alias.ref)

ROOTTEST_COMPILE_MACRO(loadcode.C
                       FIXTURES_SETUP root-treeformula-sync-loadcode-fixture)

ROOTTEST_ADD_TEST(load
                  MACRO runload.C
                  OUTREF load.ref
                  FIXTURES_REQUIRED root-treeformula-sync-loadcode-fixture)

ROOTTEST_ADD_TEST(ScanFail
                  MACRO execScanFail.C
                  COPY_TO_BUILDDIR run00155-short.root
                  OUTREF execScanFail.ref)

ROOTTEST_ADD_TEST(writefile
                  COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent 10 1 0 1 600
                  FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                  FIXTURES_SETUP root-treeformula-sync-writefile-fixture)

ROOTTEST_ADD_TEST(readfile-0
                  MACRO sync.C
                  MACROARG 0
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_SOURCE_DIR}/roottest/root/io/event\")"
                               -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                               -e "(void) gSystem->Load(\"libTestIoEvent\")"
                  OUTREF readfile-0.ref
                  WILLFAIL
                  FIXTURES_REQUIRED root-treeformula-sync-writefile-fixture)

ROOTTEST_ADD_TEST(readfile-1
                  MACRO sync.C
                  MACROARG 1
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_SOURCE_DIR}/roottest/root/io/event\")"
                               -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                               -e "(void) gSystem->Load(\"libTestIoEvent\")"
                  OUTREF readfile-1.ref
                  FIXTURES_REQUIRED root-treeformula-sync-writefile-fixture)
