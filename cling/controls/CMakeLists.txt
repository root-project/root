ROOTTEST_ADD_TEST(roottest-cling-controls-runLoopbreak
                  COPY_TO_BUILDDIR loopbreak.C
                  MACRO runLoopbreak.C
                  OUTREF Loopbreak.ref
                  LABELS roottest regression cling)

