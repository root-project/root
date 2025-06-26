ROOTTEST_ADD_TEST(write-file
                  COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent 10 1 0 1 600 Event.new.split0.root
                  FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                  FIXTURES_SETUP root-treeformula-retobj-write-fixture)

ROOTTEST_ADD_TEST(retobjTest
                  MACRO runretobjTest.C
                  COPY_TO_BUILDDIR mcpool.root
                  ROOTEXE_OPTS -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event\")"
                               -e "(void) gSystem->Load(\"libTestIoEvent\")"
                  OUTREF retobjTest.ref
                  FIXTURES_REQUIRED root-treeformula-retobj-write-fixture)

ROOTTEST_ADD_TEST(enum
                  MACRO Enum.C+
                  OUTREF enum.ref)
