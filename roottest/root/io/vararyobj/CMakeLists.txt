ROOTTEST_ADD_TEST(varyingArrayRead_51508
                  MACRO runvaryingArrayRead_51508.C+
                  COPY_TO_BUILDDIR varyingArray_51508.root
                  OUTREF varyingArrayRead_51508.good)

# on the Windows there is no "tail" command, therefore one need
# to modify test logic to be able compare output log files
if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(varyingArrayWrite
                    MACRO runvaryingArrayWrite.C+
                    OUTCNVCMD tail -120
                    FIXTURES_SETUP root-io-vararyobj-varyingArrayWrite-fixture)

    ROOTTEST_ADD_TEST(varyingArrayRead
                    MACRO runvaryingArrayRead.C+
                    OUTCNVCMD tail -120
                    FIXTURES_REQUIRED root-io-vararyobj-varyingArrayWrite-fixture
                    FIXTURES_SETUP root-io-vararyobj-varyingArrayRead-fixture)

    # compare last 120 lines from writing and reading
    ROOTTEST_ADD_TEST(varyingArrayCheck
                    COMMAND diff -b varyingArrayWrite.log varyingArrayRead.log
                    FIXTURES_REQUIRED root-io-vararyobj-varyingArrayWrite-fixture
                                        root-io-vararyobj-varyingArrayRead-fixture)


    ROOTTEST_ADD_TEST(varyingArrayNoClassDefWrite
                    MACRO runvaryingArrayNoClassDefWrite.C+
                    OUTCNVCMD tail -120
                    FIXTURES_SETUP root-io-vararyobj-varyingArrayNoClassDefWrite-fixture)

    ROOTTEST_ADD_TEST(varyingArrayNoClassDefRead
                    MACRO runvaryingArrayNoClassDefRead.C+
                    OUTCNVCMD tail -120
                    FIXTURES_REQUIRED root-io-vararyobj-varyingArrayNoClassDefWrite-fixture
                    FIXTURES_SETUP root-io-vararyobj-varyingArrayNoClassDefRead-fixture)

    # compare last 120 lines from writing and reading
    ROOTTEST_ADD_TEST(varyingArrayNoClassDefCheck
                    COMMAND diff -b varyingArrayNoClassDefWrite.log varyingArrayNoClassDefRead.log
                    FIXTURES_REQUIRED root-io-vararyobj-varyingArrayNoClassDefWrite-fixture
                                        root-io-vararyobj-varyingArrayNoClassDefRead-fixture)
endif()

ROOTTEST_ADD_TEST(writevararypolyp
                  MACRO runwritevararypolyp.C+
                  FIXTURES_SETUP root-io-vararyobj-writevararypolyp-fixture)

ROOTTEST_ADD_TEST(readvararypolyp
                  MACRO runreadvararypolyp.C+
                  OUTREF readvararypolyp.good
                  FIXTURES_REQUIRED root-io-vararyobj-writevararypolyp-fixture)

# issue #22895
ROOTTEST_GENERATE_DICTIONARY(StreamerLoopMemberwiseDict
                    ${CMAKE_CURRENT_SOURCE_DIR}/StreamerLoopMemberwise.hxx
                    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/StreamerLoopMemberwiseLinkDef.h
                    FIXTURES_SETUP root-io-vararyobj-StreamerLoopMemberwiseDict-fixture)

ROOTTEST_GENERATE_EXECUTABLE(StreamerLoopMemberwise
                    runStreamerLoopMemberwise.cxx StreamerLoopMemberwiseDict.cxx
                    LIBRARIES Core RIO Tree GTest::gtest GTest::gtest_main ROOT::TestSupport
                    FIXTURES_REQUIRED root-io-vararyobj-StreamerLoopMemberwiseDict-fixture
                    FIXTURES_SETUP root-io-vararyobj-StreamerLoopMemberwise-fixture)

ROOTTEST_ADD_TEST(StreamerLoopMemberwise
                    EXEC ${CMAKE_CURRENT_BINARY_DIR}/StreamerLoopMemberwise
                    FIXTURES_REQUIRED root-io-vararyobj-StreamerLoopMemberwise-fixture)
