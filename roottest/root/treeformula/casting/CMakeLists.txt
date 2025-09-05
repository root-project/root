ROOTTEST_ADD_TEST(write
                  COMMAND ${CMAKE_BINARY_DIR}/roottest/root/io/event/TestIoEvent 20 1 9 1 600 Event.root
                  FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                  FIXTURES_SETUP root-treeformula-casting-write-fixture)

ROOTTEST_COMPILE_MACRO(Simple.cxx
                       FIXTURES_SETUP root-treeformula-casting-Simple-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO Run.C
                  OUTREF Simple.ref
                  FIXTURES_REQUIRED root-treeformula-casting-write-fixture
                                    root-treeformula-casting-Simple-fixture)
