ROOTTEST_GENERATE_DICTIONARY(
  event_v2_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/Event_v2.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Event_v2_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_event_v2_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_event
  write_event.cxx event_v2_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_event_v2_dictionary
  FIXTURES_SETUP streamerfield_write_event_excutable)

ROOTTEST_ADD_TEST(write_event
                  EXEC ./write_event
                  FIXTURES_REQUIRED streamerfield_write_event_excutable
                  FIXTURES_SETUP streamerfield_written_event)

ROOTTEST_GENERATE_DICTIONARY(
  event_v3_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/Event_v3.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/Event_v3_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_event_v3_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  read_event
  read_event.cxx event_v3_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_event_v3_dictionary
  FIXTURES_SETUP streamerfield_read_event_executable)

ROOTTEST_ADD_TEST(read_event
                  EXEC ./read_event
                  FIXTURES_REQUIRED streamerfield_read_event_executable streamerfield_written_event)
