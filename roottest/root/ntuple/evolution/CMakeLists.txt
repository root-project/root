ROOTTEST_GENERATE_DICTIONARY(
  ntplevolv_v2_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/NtplEvolv_v2.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/NtplEvolv_v2_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_ntplevolv_v2_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_ntplevolv
  write_ntplevolv.cxx ntplevolv_v2_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_ntplevolv_v2_dictionary
  FIXTURES_SETUP write_ntplevolv_excutable)

ROOTTEST_ADD_TEST(write_ntplevolv
                  EXEC ./write_ntplevolv
                  FIXTURES_REQUIRED write_ntplevolv_excutable
                  FIXTURES_SETUP write_ntplevolv_done)

ROOTTEST_GENERATE_DICTIONARY(
  ntplevolv_v3_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/NtplEvolv_v3.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/NtplEvolv_v3_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_ntplevolv_v3_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  read_ntplevolv
  read_ntplevolv.cxx ntplevolv_v3_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_ntplevolv_v3_dictionary
  FIXTURES_SETUP read_ntplevolv_excutable)

ROOTTEST_ADD_TEST(read_ntplevolv
                  EXEC ./read_ntplevolv
                  FIXTURES_REQUIRED read_ntplevolv_excutable write_ntplevolv_done)
