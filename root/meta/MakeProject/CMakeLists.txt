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

ROOTTEST_GENERATE_DICTIONARY(stl_makeproject_test stl_makeproject_test.h LINKDEF stl_makeproject_test_linkdef.h)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-stltest-create-examples
                  COPY_TO_BUILDDIR stl_makeproject_test.h
                  MACRO create_makeproject_examples.C
                  PRECMD cmake -E remove stl_example.root || :
                  OUTREF create_makeproject_examples.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-stltest
                  MACRO runstltest.C
                  OUTREF teststlCTEST.ref
                  DEPENDS roottest-root-meta-MakeProject-stltest-create-examples)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-stltest2
                  MACRO runstltest2.C
                  DEPENDS roottest-root-meta-MakeProject-stltest)

