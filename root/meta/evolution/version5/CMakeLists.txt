ROOTTEST_ADD_TEST(roottest-root-meta-evolution-version5-execNestedColl
                  COPY_TO_BUILDDIR nestedColl.root
                  MACRO execNestedColl.cxx+
                  OUTREF execNestedColl.ref)
