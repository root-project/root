ROOTTEST_ADD_TEST(execQualifiedNames
                  MACRO execQualifiedNames.C
                  COPY_TO_BUILDDIR qualifiedName_v5.root qualifiedName_v6.root
                  OUTREF execQualifiedNames.ref
                  LABELS roottest regression io)

