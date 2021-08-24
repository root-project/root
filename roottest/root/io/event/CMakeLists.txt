#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOT_GENERATE_DICTIONARY(G__IoEventGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOT_LINKER_LIBRARY(IoEventGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/IoEventGeneration.cxx
  G__IoEventGeneration.cxx
  LIBRARIES Core Tree Hist MathCore)

if(MSVC)
  add_custom_command(TARGET IoEventGeneration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoEventGeneration.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoEventGeneration.dll
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libIoEventGeneration.lib
                                     ${CMAKE_CURRENT_BINARY_DIR}/libIoEventGeneration.lib)
endif()

ROOTTEST_GENERATE_EXECUTABLE(IoEventGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/IoEventGenerator.cxx
  LIBRARIES Core RIO Net Tree Hist MathCore IoEventGeneration
  DEPENDS IoEventGeneration
  FIXTURES_SETUP io-event-generator)

ROOTTEST_ADD_TEST(write
  COMMAND ./IoEventGenerator 10 1 1 1
  FIXTURES_REQUIRED io-event-generator
  RESOURCE_LOCK io-event-file)

ROOTTEST_ADD_TEST(read
  COMMAND ./IoEventGenerator 10 1 1 20
  FIXTURES_REQUIRED io-event-generator
  RESOURCE_LOCK io-event-file)
