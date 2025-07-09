# Reproducer of https://github.com/root-project/root/issues/18782

set(MYPARTICLE_DICTNAME MyParticleDict)
ROOTTEST_GENERATE_DICTIONARY(
	${MYPARTICLE_DICTNAME}
	${CMAKE_CURRENT_SOURCE_DIR}/MyParticle.hxx
    ${CMAKE_CURRENT_SOURCE_DIR}/MyParticle.cxx
	LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/MyParticleLinkDef.hxx
    FIXTURES_SETUP myparticle_dict_setup
)

ROOTTEST_GENERATE_EXECUTABLE(
    runtest
    runtest.cxx
    ${MYPARTICLE_DICTNAME}.cxx
    LIBRARIES Core RIO Tree GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    FIXTURES_REQUIRED myparticle_dict_setup
    FIXTURES_SETUP myparticle_runtest_setup)

ROOTTEST_ADD_TEST(runtest
    EXEC ${CMAKE_CURRENT_BINARY_DIR}/runtest
    FIXTURES_REQUIRED myparticle_runtest_setup
    COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/MyParticle.hxx)
