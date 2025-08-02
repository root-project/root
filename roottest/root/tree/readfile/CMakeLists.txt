ROOTTEST_ADD_TEST(ReadFile
                  MACRO runReadFile.C
                  COPY_TO_BUILDDIR data.txt arraydata.txt arraydata2.txt data.csv spectrum.csv
                  OUTREF ReadFile.ref)