ROOTTEST_ADD_TEST(LoadDim
                  MACRO execLoadDim.C
                  COPY_TO_BUILDDIR loadDim.root
                  OUTREF execLoadDim.ref)

ROOTTEST_ADD_TEST(missingString
                  MACRO execmissingString.C
                  COPY_TO_BUILDDIR missingString.root
                  OUTREF execmissingString.ref)

ROOTTEST_ADD_TEST(FormattingWrite
                  MACRO runFormatting.C
                  MACROARG 1
                  FIXTURES_SETUP root-treeformula-scan-FormattingWrite-fixtures)

ROOTTEST_ADD_TEST(Formatting
                  MACRO runFormatting.C
                  OUTREF Formatting.ref
                  FIXTURES_REQUIRED root-treeformula-scan-FormattingWrite-fixtures)
