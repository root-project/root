ROOTTEST_COMPILE_MACRO(single.C
                       FIXTURES_SETUP root-aclic-nolinkdep-single_C-fixture)

ROOTTEST_COMPILE_MACRO(script1.C
                       FIXTURES_SETUP root-aclic-nolinkdep-script1_C-fixture)

ROOTTEST_COMPILE_MACRO(script2.C
                       BUILDLIB single_C
                       FIXTURES_REQUIRED root-aclic-nolinkdep-single_C-fixture
                       FIXTURES_SETUP root-aclic-nolinkdep-script2_C-fixture)

ROOTTEST_ADD_TEST(single
                  MACRO single.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-single_C-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-single-fixture)

ROOTTEST_ADD_TEST(load1
                  MACRO script1.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-script1_C-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load1-fixture)

ROOTTEST_ADD_TEST(load2
                  MACRO script2.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-script2_C-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load2-fixture)

ROOTTEST_ADD_TEST(load2b
                  MACRO script2.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load2-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load2b-fixture)

ROOTTEST_ADD_TEST(load1b
                  MACRO script1.C+
                  OUTREF twoscript.ref
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load1-fixture
                                    root-aclic-nolinkdep-load2b-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-load1b-fixture)

# after load1b plain tests are finished and now try to rename script2.C to script.C and run it
ROOTTEST_ADD_TEST(build10
                  MACRO link.C
                  PRECMD ${CMAKE_COMMAND} -E remove "single_C.* script_C.*"
                  COPY_TO_BUILDDIR single.C script.C script2.C
                  FIXTURES_REQUIRED root-aclic-nolinkdep-load1b-fixture
                  FIXTURES_SETUP root-aclic-nolinkdep-build10-fixture)

ROOTTEST_ADD_TEST(load10
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/script.C+
                  FIXTURES_REQUIRED root-aclic-nolinkdep-build10-fixture)
