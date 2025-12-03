ROOTTEST_ADD_TEST(execNestedColl
                  COPY_TO_BUILDDIR nestedColl.root
                  MACRO execNestedColl.cxx+
                  OUTREF execNestedColl.ref)
