ROOTTEST_ADD_TEST(leaflist
                  MACRO runleaflist.C
                  COPY_TO_BUILDDIR badleafc.root
                  OUTREF leaflist.ref)

ROOTTEST_ADD_TEST(ReadFile
                  MACRO runReadFile.C
                  COPY_TO_BUILDDIR shortstring.txt longstring.txt
                  OUTREF ReadFile.ref)