if(NOT MSVC)
  ROOTTEST_GENERATE_REFLEX_DICTIONARY(libUser
                                      User.hpp
                                      SELECTION User.xml
                                      LIBNAME libUser
                                      NO_ROOTMAP
                                      FIXTURES_SETUP root-meta-race-12552-fixture)

  ROOTTEST_ADD_TEST(dlopenrace
                    MACRO exec_dlopenrace.C
                    FIXTURES_REQUIRED root-meta-race-12552-fixture)
endif()

