ROOTTEST_GENERATE_DICTIONARY(stl_makeproject_test
                             stl_makeproject_test.h
                             LINKDEF stl_makeproject_test_linkdef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-MakeProject-stl_makeproject_test-fixture)
# If we build roottest in-tree, add a dependecy on Hist because the header
# includes TH1D.
if(TARGET Hist)
  add_dependencies(stl_makeproject_test Hist)
endif()

if(runtime_cxxmodules)
  # FIXME: For C++ modules builds, module.modulemap is generated during configuration time
  # and it is not needed during execution of stltest*, but later for other tests it needs to be back in build directory.
  # FIXME: We need to run some tests in serial mode, because in very-parallel builds,
  # since they access the same build directory, they could conflict via existent/not module.modulemap file.
  ROOTTEST_ADD_TEST(runatlasaod
                  COPY_TO_BUILDDIR enable_modulemap.sh
                  PRECMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  COPY_TO_BUILDDIR small_aod.pool.root
                  MACRO runatlasaod.C
                  OUTCNV atlasaod_convert.sh
                  OUTREF atlasaod.ref
                  LABELS longtest
                  RUN_SERIAL)

    ROOTTEST_ADD_TEST(runcms310
                  COPY_TO_BUILDDIR enable_modulemap.sh
                  PRECMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  COPY_TO_BUILDDIR CMSSW_3_1_0_pre11-RelValZTT-default-copy.root
                  MACRO runcms310.C
                  OUTCNV cms310_convert.sh
                  OUTREF cms310.ref
                  LABELS longtest
                  RUN_SERIAL)

    ROOTTEST_ADD_TEST(runaliceesd
                  COPY_TO_BUILDDIR enable_modulemap.sh
                  PRECMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  COPY_TO_BUILDDIR alice_ESDs.root
                  MACRO runaliceesd.C
                  OUTCNV aliceesd_convert.sh
                  OUTREF aliceesd.ref
                  RUN_SERIAL)

    ROOTTEST_ADD_TEST(examples
                  COPY_TO_BUILDDIR enable_modulemap.sh disable_modulemap.sh stl_makeproject_test.h
                  PRECMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  MACRO create_makeproject_examples.C
                  OUTREF create_makeproject_examples.ref
                  FIXTURES_REQUIRED root-meta-MakeProject-stl_makeproject_test-fixture
                  FIXTURES_SETUP root-meta-MakeProject-examples-fixture)

    ROOTTEST_ADD_TEST(stltest
                  PRECMD ./disable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  MACRO runstltest.C
                  POSTCMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  OUTREF teststlCTEST.ref
                  FIXTURES_REQUIRED root-meta-MakeProject-examples-fixture
                  FIXTURES_SETUP root-meta-MakeProject-stltest-fixture)

    ROOTTEST_ADD_TEST(stltest2
                  PRECMD ./disable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  MACRO runstltest2.C
                  POSTCMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  OUTREF stltest2.ref
                  FIXTURES_REQUIRED root-meta-MakeProject-stltest-fixture)

    ROOTTEST_ADD_TEST(foreign
                  COPY_TO_BUILDDIR enable_modulemap.sh foreign.root
                  PRECMD ./enable_modulemap.sh ${CMAKE_CURRENT_BINARY_DIR}
                  MACRO runforeign.C
                  OUTREF foreign.ref
                  RUN_SERIAL)
else()
    ROOTTEST_ADD_TEST(runatlasaod
                  COPY_TO_BUILDDIR small_aod.pool.root
                  MACRO runatlasaod.C
                  OUTCNV atlasaod_convert.sh
                  OUTREF atlasaod.ref
                  LABELS longtest)

    ROOTTEST_ADD_TEST(runcms310
                  COPY_TO_BUILDDIR CMSSW_3_1_0_pre11-RelValZTT-default-copy.root
                  MACRO runcms310.C
                  OUTCNV cms310_convert.sh
                  OUTREF cms310.ref
                  LABELS longtest)

    ROOTTEST_ADD_TEST(runaliceesd
                  COPY_TO_BUILDDIR alice_ESDs.root
                  MACRO runaliceesd.C
                  OUTCNV aliceesd_convert.sh
                  OUTREF aliceesd.ref)

    ROOTTEST_ADD_TEST(examples
                  COPY_TO_BUILDDIR stl_makeproject_test.h
                  MACRO create_makeproject_examples.C
                  OUTREF create_makeproject_examples.ref
                  FIXTURES_REQUIRED root-meta-MakeProject-stl_makeproject_test-fixture
                  FIXTURES_SETUP root-meta-MakeProject-examples-fixture)

    ROOTTEST_ADD_TEST(stltest
                  MACRO runstltest.C
                  OUTREF teststlCTEST${ref_suffix}
                  FIXTURES_REQUIRED root-meta-MakeProject-examples-fixture
                  FIXTURES_SETUP root-meta-MakeProject-stltest-fixture)

    ROOTTEST_ADD_TEST(stltest2
                  MACRO runstltest2.C
                  OUTREF stltest2.ref
                  FIXTURES_REQUIRED root-meta-MakeProject-stltest-fixture)

    ROOTTEST_ADD_TEST(foreign
                      COPY_TO_BUILDDIR foreign.root
                      MACRO runforeign.C
                      OUTREF foreign${ref_suffix})
endif()

