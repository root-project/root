# building shared library with missing symbols does not work on windows
# probably one can manipulate linker flags via rootlogon.C script - as it done for Linux and Mac
if(MSVC AND NOT win_broken_tests)
  return()
endif()

ROOTTEST_ADD_TEST(build1
                  MACRO link.C
                  PRECMD ${CMAKE_COMMAND} -E remove "script1_C.* script2_C.*"
                  COPY_TO_BUILDDIR single.C script.C script1.C script2.C rootlogon.C
                  FIXTURES_SETUP root-aclic-nolinkdep-build1-fixture)

ROOTTEST_ADD_TEST(single
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/single.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-build1-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-single-fixture)


ROOTTEST_ADD_TEST(load1
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script1.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-build1-fixture
                                    root-aclic-nolinkdep-single-fixture 
                  FIXTURES_SETUP root-aclic-nolinkdep-load1-fixture)

ROOTTEST_ADD_TEST(load2
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script2.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-build1-fixture
                                    root-aclic-nolinkdep-single-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load2-fixture)

ROOTTEST_ADD_TEST(load2b
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script2.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load2-fixture
                                    root-aclic-nolinkdep-single-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load2b-fixture)

ROOTTEST_ADD_TEST(load1b
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script1.C+
                  PRECMD ${CMAKE_COMMAND} -E remove "single_C.*"
                  OUTREF twoscript.ref
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load1-fixture
                                    root-aclic-nolinkdep-load2b-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load1b-fixture)

ROOTTEST_ADD_TEST(build10
                  MACRO link.C
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load1b-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-build10-fixture)

ROOTTEST_ADD_TEST(load10
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-build10-fixture)
