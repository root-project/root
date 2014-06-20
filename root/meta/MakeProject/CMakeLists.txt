ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runatlasaod
                  COPY_TO_BUILDDIR small_aod.pool.root
                  MACRO runatlasaod.C
                  OUTCNV atlasaod_convert.sh
                  OUTREF atlasaod.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runcms310
                  COPY_TO_BUILDDIR CMSSW_3_1_0_pre11-RelValZTT-default-copy.root
                  MACRO runcms310.C
                  OUTCNV cms310_convert.sh
                  OUTREF cms310.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-runaliceesd
                  COPY_TO_BUILDDIR alice_ESDs.root
                  MACRO runaliceesd.C
                  OUTCNV aliceesd_convert.sh
                  OUTREF aliceesd.ref)
