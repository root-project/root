ROOT_GENERATE_DICTIONARY(G__IoBigEventGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOT_LINKER_LIBRARY(IoBigEventGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/IoBigEventGeneration.cxx
  G__IoBigEventGeneration.cxx
  LIBRARIES Core Tree Hist MathCore Physics Graf Matrix)

if(MSVC)
  add_custom_command(TARGET IoBigEventGeneration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoBigEventGeneration.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoBigEventGeneration.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoBigEventGeneration.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoBigEventGeneration.lib)
endif()

ROOTTEST_GENERATE_EXECUTABLE(IoBigEventGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/IoBigEventGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore IoBigEventGeneration
  DEPENDS IoBigEventGeneration MathCore
  FIXTURES_SETUP io-bigevent-generator)

ROOTTEST_ADD_TEST(write
  COMMAND ./IoBigEventGenerator 10 1 1 1
  FIXTURES_REQUIRED io-bigevent-generator
  RESOURCE_LOCK io-bigevent-file)

ROOTTEST_ADD_TEST(read
  COMMAND ./IoBigEventGenerator 10 1 1 20
  FIXTURES_REQUIRED io-bigevent-generator
  RESOURCE_LOCK io-bigevent-file)
