ROOTTEST_GENERATE_DICTIONARY(
  test_auto_ptr_v2_dict
  TestAutoPtr_v2.hxx
  LINKDEF TestAutoPtr_v2_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP root-io-autoptr-dictv2-fixture
)

ROOTTEST_GENERATE_EXECUTABLE(
  write_autoptr
  write_autoptr.cxx test_auto_ptr_v2_dict.cxx
  LIBRARIES Core RIO Tree
  FIXTURES_REQUIRED root-io-autoptr-dictv2-fixture
  FIXTURES_SETUP root-io-autoptr-write_autoptr_executable-fixture)

ROOTTEST_ADD_TEST(write_autoptr
                  EXEC ./write_autoptr
                  FIXTURES_REQUIRED root-io-autoptr-write_autoptr_executable-fixture
                  FIXTURES_SETUP root-io-autoptr-write_autoptr-fixture)

ROOTTEST_GENERATE_DICTIONARY(
  test_auto_ptr_v3_dict
  TestAutoPtr_v3.hxx
  LINKDEF TestAutoPtr_v3_LinkDef.h
  NO_ROOTMAP NO_CXXMODULE
  FIXTURES_SETUP root-io-autoptr-dictv3-fixture
)

ROOTTEST_GENERATE_EXECUTABLE(
  evolve_autoptr
  evolve_autoptr.cxx test_auto_ptr_v3_dict.cxx
  LIBRARIES Core RIO Tree
  FIXTURES_REQUIRED root-io-autoptr-dictv3-fixture
  FIXTURES_SETUP root-io-autoptr-evolve_autoptr_executable-fixture)

ROOTTEST_ADD_TEST(evolve_autoptr
                  EXEC ./evolve_autoptr
                  FIXTURES_REQUIRED root-io-autoptr-evolve_autoptr_executable-fixture
                                    root-io-autoptr-write_autoptr-fixture)
