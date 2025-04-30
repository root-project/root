ROOTTEST_GENERATE_DICTIONARY(
  event_v2_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/Event_v2.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Event_v2_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_event_v2_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_event
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_event_v2_dictionary
  FIXTURES_SETUP write_event_excutable)

target_sources(
  write_event
  PRIVATE write_event.cxx event_v2_dict.cxx
)

ROOTTEST_ADD_TEST(write_event
                  EXEC ./write_event
                  FIXTURES_REQUIRED write_event_excutable
                  FIXTURES_SETUP written_event)

ROOTTEST_GENERATE_DICTIONARY(
  event_v3_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/Event_v3.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Event_v3_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_event_v3_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  read_event
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_event_v3_dictionary
  FIXTURES_SETUP read_event_executable)

target_sources(
  read_event
  PRIVATE read_event.cxx event_v3_dict.cxx
)

ROOTTEST_ADD_TEST(read_event
                  EXEC ./read_event
                  FIXTURES_REQUIRED read_event_executable written_event)
