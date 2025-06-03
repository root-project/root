ROOTTEST_GENERATE_DICTIONARY(libIoBigEventGeneration
  Event.h
  LINKDEF EventLinkDef.h
  SOURCES IoBigEventGeneration.cxx
  NO_ROOTMAP
  NO_CXXMODULE
  FIXTURES_SETUP io-bigevent-dict)

target_link_libraries(libIoBigEventGenerationlibgen Physics)

ROOTTEST_GENERATE_EXECUTABLE(IoBigEventGenerator
  IoBigEventGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore libIoBigEventGenerationlibgen
  FIXTURES_REQUIRED io-bigevent-dict
  FIXTURES_SETUP io-bigevent-generator)

ROOTTEST_ADD_TEST(write
  COMMAND ./IoBigEventGenerator 10 1 1 1
  FIXTURES_REQUIRED io-bigevent-generator
  RESOURCE_LOCK io-bigevent-file)

ROOTTEST_ADD_TEST(read
  COMMAND ./IoBigEventGenerator 10 1 1 20
  FIXTURES_REQUIRED io-bigevent-generator
  RESOURCE_LOCK io-bigevent-file)
