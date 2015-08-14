if (NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

configure_file(test.root . COPYONLY)
configure_file(simpleTree.root . COPYONLY)

############################## PATTERN TESTS ############################
set (TESTPATTERN_EXE ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py)
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern1
                  COMMAND ${TESTPATTERN_EXE} test.root
                  OUTREF SimplePattern.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern2
                  COMMAND ${TESTPATTERN_EXE} test.root:tof
                  OUTREF SimplePattern2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern3
                  COMMAND ${TESTPATTERN_EXE} test.root:*
                  OUTREF SimplePattern3.ref)
#########################################################################

set(TOOLS_PREFIX ${PYTHON_EXECUTABLE} ${ROOTSYS}/bin)

############################## ROOLS TESTS ##############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools1
                  COMMAND ${TOOLS_PREFIX}/rools test.root
                  OUTREF SimpleRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools2
                  COMMAND ${TOOLS_PREFIX}/rools -1 test.root:*
                  OUTREF SimpleRools2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools3
                  COMMAND ${TOOLS_PREFIX}/rools -1 test.root:tof
                  OUTREF SimpleRools3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools4
                  COMMAND ${TOOLS_PREFIX}/rools -1 test.root:tof/*
                  OUTREF SimpleRools4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools1
                  COMMAND ${TOOLS_PREFIX}/rools -l test.root
                  OUTREF LongRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools2
                  COMMAND ${TOOLS_PREFIX}/rools -l test.root:*
                  OUTREF LongRools2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools3
                  COMMAND ${TOOLS_PREFIX}/rools -l test.root:tof
                  OUTREF LongRools3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools4
                  COMMAND ${TOOLS_PREFIX}/rools -l test.root:tof/*
                  OUTREF LongRools4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRools1
                  COMMAND ${TOOLS_PREFIX}/rools http://root.cern.ch/files/pippa.root
                  OUTREF WebRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRools2
                  COMMAND ${TOOLS_PREFIX}/rools -l http://root.cern.ch/files/pippa.root
                  OUTREF WebRools2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-TreeRools1
                  COMMAND ${TOOLS_PREFIX}/rools -t simpleTree.root
                  OUTREF TreeRooLs.ref)

#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root victim1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1
                  COMMAND ${TOOLS_PREFIX}/roorm victim1.root:hpx
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools victim1.root
                  OUTREF SimpleRoorm.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root victim2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim2.root:tof/plane0
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 victim2.root:tof
                  OUTREF SimpleRoorm2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput)

#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1
                  COMMAND ${TOOLS_PREFIX}/roomkdir target1.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools target1.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2
                  COMMAND ${TOOLS_PREFIX}/roomkdir target2.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools target2.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3
                  COMMAND ${TOOLS_PREFIX}/roomkdir target3.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools target3.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4
                  COMMAND ${TOOLS_PREFIX}/roomkdir target4.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools target4.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target4.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4CheckOutput)
#########################################################################

endif() # these tests are not run with the classic build
