ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runatlasaod
                  MACRO runatlasaod.C
                  OUTCNV atlasaod_convert.sh
                  OUTREF atlasaod.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runcms310
                  MACRO runcms310.C
                  OUTCNV cms310_convert.sh
                  OUTREF cms310.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runaliceesd
                  MACRO runaliceesd.C
                  OUTCNV aliceesd_convert.sh
                  OUTREF aliceesd.ref)
