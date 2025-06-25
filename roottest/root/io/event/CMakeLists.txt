ROOTTEST_GENERATE_DICTIONARY(libTestIoEvent
                             Event.h
                             LINKDEF EventLinkDef.h
                             SOURCES Event.cxx
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-event-libTestIoEvent-fixture)

ROOTTEST_GENERATE_EXECUTABLE(TestIoEvent
                             MainEvent.cxx
                             LIBRARIES Core RIO libTestIoEventlibgen
                             FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture
                             FIXTURES_SETUP root-io-event-TestIoEvent-fixture)

ROOTTEST_ADD_TEST(write
                  COMMAND ./TestIoEvent 10 1 1 1
                  FIXTURES_REQUIRED root-io-event-TestIoEvent-fixture
                  FIXTURES_SETUP root-io-event-write-fixture)

ROOTTEST_ADD_TEST(read
                  COMMAND ./TestIoEvent 10 1 1 20
                  FIXTURES_REQUIRED root-io-event-write-fixture)
