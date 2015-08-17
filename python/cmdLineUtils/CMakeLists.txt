if (ROOT_python_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

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
                  COMMAND ${TOOLS_PREFIX}/rools -1 test.root
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
                  COMMAND ${TOOLS_PREFIX}/rools -1 http://root.cern.ch/files/pippa.root
                  OUTREF WebRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRools2
                  COMMAND ${TOOLS_PREFIX}/rools -1 http://root.cern.ch/files/pippa.root:LL
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
                  COMMAND ${TOOLS_PREFIX}/rools -1 victim1.root
                  OUTREF SimpleRoorm.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root victim2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2
                  COMMAND ${TOOLS_PREFIX}/roorm victim2.root:hp*
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 victim2.root
                  OUTREF SimpleRoorm2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root victim3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm3
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim3.root:tof/plane0
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 victim3.root:tof
                  OUTREF SimpleRoorm3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm3Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r victim3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm3CheckOutput)
#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1
                  COMMAND ${TOOLS_PREFIX}/roomkdir target1.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 target1.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2
                  COMMAND ${TOOLS_PREFIX}/roomkdir target2.root:dir1 target2.root:dir2
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 target2.root
                  OUTREF SimpleRoomkdir2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root target3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3
                  COMMAND ${TOOLS_PREFIX}/roomkdir -p target3.root:aa/bb/cc
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 target3.root:aa/bb
                  OUTREF SimpleRoomkdir3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r target3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput)
#########################################################################

############################# ROOCP TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root copy1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp1
                  COMMAND ${TOOLS_PREFIX}/roocp copy1.root:hpx copy1.root:histo
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 copy1.root
                  OUTREF SimpleRoocp.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r copy1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root copy2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp2
                  COMMAND ${TOOLS_PREFIX}/roocp -r copy2.root:tof copy2.root:fot
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 copy2.root
                  OUTREF SimpleRoocp2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r copy2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root copy3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp3
                  COMMAND ${TOOLS_PREFIX}/roocp --replace copy3.root:hpx copy3.root:hpxpy
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 copy3.root
                  OUTREF SimpleRoocp3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp3Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r copy3.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRoocp3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root copy4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp4
                  COMMAND ${TOOLS_PREFIX}/roocp copy4.root:hpx copy4.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 copy4.root:dir
                  OUTREF SimpleRoocp4.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp4Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r copy4.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRoocp4CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root copy5.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp5
                  COMMAND ${TOOLS_PREFIX}/roocp -r copy5.root:tof copy5.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp5PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 copy5.root:dir
                  OUTREF SimpleRoocp5.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp5)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoocp5Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r copy5.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoocp5CheckOutput)
#########################################################################

############################# ROOMV TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root move1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv1
                  COMMAND ${TOOLS_PREFIX}/roomv move1.root:hpx move1.root:histo
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 move1.root
                  OUTREF SimpleRoomv.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv1Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r move1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root move2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv2
                  COMMAND ${TOOLS_PREFIX}/roomv move2.root:tof move2.root:fot
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 move2.root
                  OUTREF SimpleRoomv2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv2Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r move2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root move3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv3
                  COMMAND ${TOOLS_PREFIX}/roomv move3.root:hpx move3.root:hpxpy
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 move3.root
                  OUTREF SimpleRoomv3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv3Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r move3.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRoomv3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root move4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv4
                  COMMAND ${TOOLS_PREFIX}/roomv move4.root:hpx move4.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 move4.root:*
                  OUTREF SimpleRoomv4.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv4Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r move4.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRoomv4CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/roocp -r test.root move5.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv5
                  COMMAND ${TOOLS_PREFIX}/roomv move5.root:tof move5.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv5PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rools -1 move5.root:*
                  OUTREF SimpleRoomv5.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv5)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomv5Clean
                  COMMAND ${TOOLS_PREFIX}/roorm -r move5.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomv5CheckOutput)
#########################################################################

endif() # these tests are not run with the classic build
