ROOTTEST_GENERATE_DICTIONARY(IoEventDict Event.h
  LINKDEF EventLinkDef.h
  FIXTURES_SETUP io-event-dict)

target_sources(IoEventDictlibgen PRIVATE Event.cxx)

ROOTTEST_GENERATE_EXECUTABLE(IoEventGenerator
  MainEvent.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore IoEventDictlibgen
  FIXTURES_REQUIRED io-event-dict
  FIXTURES_SETUP io-event-generator)

ROOTTEST_ADD_TEST(write
  COMMAND ./IoEventGenerator 10 1 1 1
  FIXTURES_REQUIRED io-event-generator
  FIXTURES_SETUP io-event-write
  RESOURCE_LOCK io-event-file)

ROOTTEST_ADD_TEST(read
  COMMAND ./IoEventGenerator 10 1 1 20
  FIXTURES_REQUIRED io-event-write
  RESOURCE_LOCK io-event-file)
