ROOTTEST_ADD_TEST(simple
                  MACRO runcstring.C
                  COPY_TO_BUILDDIR rootlogon.C cstring.C
                  OUTREF cstring.ref)
