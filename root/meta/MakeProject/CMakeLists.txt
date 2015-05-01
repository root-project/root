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

ROOTTEST_GENERATE_DICTIONARY(stl_makeproject_test LINKDEF stl_makeproject_test.h stl_makeproject_test_linkdef.h)
set(depends ${depends} ${GENERATE_DICTIONARY_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-create-examples
                  MACRO create_makeproject_examples.C
                  PRECMD cmake -E remove stl_example.root || :
                  OUTREF create_makeproject_examples.ref
                  DEPENDS ${depends})

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-cleanup
                  COMMAND cmake -E remove stl_makeproject_test.so stl_makeproject_test.rootmap stl_makeproject_test_rdict.pcm stl_makeproject_test.cxx stl_example.root
                  DEPENDS roottest-root-meta-MakeProject-create-examples)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-stltest
                  MACRO runstltest.C
                  OUTREF teststl.ref
                  DEPENDS roottest-root-meta-MakeProject-cleanup)

ROOTTEST_ADD_TEST(roottest-root-meta-MakeProject-stltest2
                  MACRO runstltest2.C+
                  DEPENDS roottest-root-meta-MakeProject-stltest)

