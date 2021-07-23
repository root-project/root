ROOTTEST_ADD_TEST(runLoopbreak
                  COPY_TO_BUILDDIR loopbreak.C
                  MACRO runLoopbreak.C
                  OUTREF Loopbreak.ref
                  LABELS roottest regression cling)
