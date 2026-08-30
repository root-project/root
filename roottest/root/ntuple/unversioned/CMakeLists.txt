ROOTTEST_GENERATE_DICTIONARY(
  ntplhit_old_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/NtplHit_Old.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/NtplHit_Old_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_ntplhit_old_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_ntplhit
  write_ntplhit.cxx ntplhit_old_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_ntplhit_old_dictionary
  FIXTURES_SETUP write_ntplhit_excutable)

ROOTTEST_ADD_TEST(write_ntplhit
                  EXEC ./write_ntplhit
                  FIXTURES_REQUIRED write_ntplhit_excutable
                  FIXTURES_SETUP write_ntplhit_done)

ROOTTEST_GENERATE_DICTIONARY(
  ntplhit_new_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/NtplHit_New.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/NtplHit_New_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_ntplhit_new_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  read_ntplhit
  read_ntplhit.cxx ntplhit_new_dict.cxx
  LIBRARIES Core RIO ROOTNTuple
  FIXTURES_REQUIRED generated_ntplhit_new_dictionary
  FIXTURES_SETUP read_ntplhit_excutable)

ROOTTEST_ADD_TEST(read_ntplhit
                  EXEC ./read_ntplhit
                  FIXTURES_REQUIRED read_ntplhit_excutable write_ntplhit_done)
