ROOTTEST_ADD_TEST(zombie
                  COPY_TO_BUILDDIR crashed.root
                  MACRO assertZombie.C
                  ERRREF assertZombie.ref)

ROOTTEST_ADD_TEST(empty
                  MACRO runempty.C
                  OUTREF empty.ref)
