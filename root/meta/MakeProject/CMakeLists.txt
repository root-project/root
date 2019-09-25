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

ROOTTEST_GENERATE_DICTIONARY(stl_makeproject_test stl_makeproject_test.h LINKDEF stl_makeproject_test_linkdef.h)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/module.modulemap")
  # For C++ modules builds, module.modulemap is generated during configuration time.
  ROOTTEST_ADD_TEST(examples
                   COPY_TO_BUILDDIR stl_makeproject_test.h
                   MACRO create_makeproject_examples.C
                   POSTCMD ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/module.modulemap ${CMAKE_CURRENT_BINARY_DIR}/disabled.module.modulemap
                   OUTREF create_makeproject_examples.ref
                   DEPENDS ${GENERATE_DICTIONARY_TEST} stl_makeproject_test-build)
elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/disabled.module.modulemap")
  # For C++ modules incremental builds.
  ROOTTEST_ADD_TEST(examples
                  COPY_TO_BUILDDIR stl_makeproject_test.h
                  PRECMD ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/disabled.module.modulemap ${CMAKE_CURRENT_BINARY_DIR}/module.modulemap
                  MACRO create_makeproject_examples.C
                  POSTCMD ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/module.modulemap ${CMAKE_CURRENT_BINARY_DIR}/disabled.module.modulemap
                  OUTREF create_makeproject_examples.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST} stl_makeproject_test-build)
elseif(NOT ROOT_runtime_cxxmodules_FOUND)
  ROOTTEST_ADD_TEST(examples
                    COPY_TO_BUILDDIR stl_makeproject_test.h
                    MACRO create_makeproject_examples.C
                    OUTREF create_makeproject_examples.ref
                    DEPENDS ${GENERATE_DICTIONARY_TEST} stl_makeproject_test-build)
endif()


ROOTTEST_ADD_TEST(stltest
                  MACRO runstltest.C
                  OUTREF teststlCTEST.ref
                  DEPENDS examples)

ROOTTEST_ADD_TEST(stltest2
                  MACRO runstltest2.C
                  DEPENDS stltest)

ROOTTEST_ADD_TEST(foreign
                  COPY_TO_BUILDDIR foreign.root
                  MACRO runforeign.C)

