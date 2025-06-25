ROOTTEST_ADD_TEST(interpreted
                  MACRO runvbase.C
                  OUTREF vbase-i.out.ref
                  ERRREF vbase-i.err.ref
                  LABELS roottest regression cling
                  PRECMD ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/runvbase_C${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/runvbase_C_ACLiC_dict_rdict.pcm
                  FIXTURES_SETUP cling-offset-runvbase-cling-fixture)

ROOTTEST_COMPILE_MACRO(runvbase.C
                       FIXTURES_REQUIRED cling-offset-runvbase-cling-fixture
                       FIXTURES_SETUP cling-offset-runvbase-compile-fixture)

ROOTTEST_ADD_TEST(compiled
                  MACRO runvbase.C+
                  OUTREF vbase-c.out.ref
                  ERRREF vbase-c.err.ref
                  LABELS roottest regression cling
                  FIXTURES_REQUIRED cling-offset-runvbase-compile-fixture)

