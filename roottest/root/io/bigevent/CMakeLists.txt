ROOTTEST_GENERATE_DICTIONARY(libIoBigEventGeneration
  Event.h
  LINKDEF EventLinkDef.h
  SOURCES IoBigEventGeneration.cxx
  NO_ROOTMAP
  NO_CXXMODULE
  FIXTURES_SETUP root-io-bigevent-dict-fixture)

target_link_libraries(libIoBigEventGenerationlibgen Physics)

ROOTTEST_GENERATE_EXECUTABLE(IoBigEventGenerator
  IoBigEventGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore libIoBigEventGenerationlibgen
  FIXTURES_REQUIRED root-io-bigevent-dict-fixture
  FIXTURES_SETUP root-io-bigevent-generator-fixture)

ROOTTEST_ADD_TEST(write
  COMMAND ./IoBigEventGenerator 10 1 1 1
  FIXTURES_REQUIRED root-io-bigevent-generator-fixture
  FIXTURES_SETUP root-io-bigevent-write-fixture)

ROOTTEST_ADD_TEST(read
  COMMAND ./IoBigEventGenerator 10 1 1 20
  FIXTURES_REQUIRED root-io-bigevent-generator-fixture root-io-bigevent-write-fixture)
