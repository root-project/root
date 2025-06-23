ROOTTEST_GENERATE_DICTIONARY(
  test_auto_ptr_v2_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/TestAutoPtr_v2.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/TestAutoPtr_v2_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_test_auto_ptr_v2_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_autoptr
  LIBRARIES Core RIO Tree
  FIXTURES_REQUIRED generated_test_auto_ptr_v2_dictionary
  FIXTURES_SETUP write_autoptr_executable)

target_sources(
  write_autoptr
  PRIVATE write_autoptr.cxx test_auto_ptr_v2_dict.cxx
)

ROOTTEST_ADD_TEST(write_autoptr
                  EXEC ./write_autoptr
                  FIXTURES_REQUIRED write_autoptr_executable
                  FIXTURES_SETUP written_autoptr)

ROOTTEST_GENERATE_DICTIONARY(
  test_auto_ptr_v3_dict
  ${CMAKE_CURRENT_SOURCE_DIR}/TestAutoPtr_v3.hxx
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/TestAutoPtr_v3_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP generated_test_auto_ptr_v3_dictionary
)

ROOTTEST_GENERATE_EXECUTABLE(
  evolve_autoptr
  LIBRARIES Core RIO Tree
  FIXTURES_REQUIRED generated_test_auto_ptr_v3_dictionary
  FIXTURES_SETUP evolve_autoptr_executable)

target_sources(
  evolve_autoptr
  PRIVATE evolve_autoptr.cxx test_auto_ptr_v3_dict.cxx
)

ROOTTEST_ADD_TEST(evolve_autoptr
                  EXEC ./evolve_autoptr
                  FIXTURES_REQUIRED evolve_autoptr_executable written_autoptr)
