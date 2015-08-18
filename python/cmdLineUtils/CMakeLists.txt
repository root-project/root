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
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls -1 test.root
                  OUTREF SimpleRootls.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls -1 test.root:*
                  OUTREF SimpleRootls2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls -1 test.root:tof
                  OUTREF SimpleRootls3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls -1 test.root:tof/*
                  OUTREF SimpleRootls4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls -l test.root
                  OUTREF LongRootls.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls -l test.root:*
                  OUTREF LongRootls2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls -l test.root:tof
                  OUTREF LongRootls3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls -l test.root:tof/*
                  OUTREF LongRootls4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls -1 http://root.cern.ch/files/pippa.root
                  OUTREF WebRootls.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls -1 http://root.cern.ch/files/pippa.root:LL
                  OUTREF WebRootls2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-TreeRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls -t simpleTree.root
                  OUTREF TreeRootls.ref)

#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root victim1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm1
                  COMMAND ${TOOLS_PREFIX}/rootrm victim1.root:hpx
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 victim1.root
                  OUTREF SimpleRootrm.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r victim1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root victim2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm2
                  COMMAND ${TOOLS_PREFIX}/rootrm victim2.root:hp*
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 victim2.root
                  OUTREF SimpleRootrm2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r victim2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root victim3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm3
                  COMMAND ${TOOLS_PREFIX}/rootrm -r victim3.root:tof/plane0
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 victim3.root:tof
                  OUTREF SimpleRootrm3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootrm3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r victim3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootrm3CheckOutput)
#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root target1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir1
                  COMMAND ${TOOLS_PREFIX}/rootmkdir target1.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 target1.root
                  OUTREF SimpleRootmkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r target1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root target2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir2
                  COMMAND ${TOOLS_PREFIX}/rootmkdir target2.root:dir1 target2.root:dir2
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 target2.root
                  OUTREF SimpleRootmkdir2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r target2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root target3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir3
                  COMMAND ${TOOLS_PREFIX}/rootmkdir -p target3.root:aa/bb/cc
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 target3.root:aa/bb
                  OUTREF SimpleRootmkdir3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmkdir3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r target3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmkdir3CheckOutput)
#########################################################################

############################# ROOCP TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root copy1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp1
                  COMMAND ${TOOLS_PREFIX}/rootcp copy1.root:hpx copy1.root:histo
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 copy1.root
                  OUTREF SimpleRootcp.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r copy1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root copy2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp2
                  COMMAND ${TOOLS_PREFIX}/rootcp -r copy2.root:tof copy2.root:fot
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 copy2.root
                  OUTREF SimpleRootcp2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r copy2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root copy3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp3
                  COMMAND ${TOOLS_PREFIX}/rootcp --replace copy3.root:hpx copy3.root:hpxpy
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 copy3.root
                  OUTREF SimpleRootcp3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r copy3.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRootcp3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root copy4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp4
                  COMMAND ${TOOLS_PREFIX}/rootcp copy4.root:hpx copy4.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 copy4.root:dir
                  OUTREF SimpleRootcp4.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp4Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r copy4.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRootcp4CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root copy5.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp5
                  COMMAND ${TOOLS_PREFIX}/rootcp -r copy5.root:tof copy5.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp5PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 copy5.root:dir
                  OUTREF SimpleRootcp5.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp5)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootcp5Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r copy5.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootcp5CheckOutput)
#########################################################################

############################# ROOMV TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root move1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv1
                  COMMAND ${TOOLS_PREFIX}/rootmv move1.root:hpx move1.root:histo
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 move1.root
                  OUTREF SimpleRootmv.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r move1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root move2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv2
                  COMMAND ${TOOLS_PREFIX}/rootmv move2.root:tof move2.root:fot
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 move2.root
                  OUTREF SimpleRootmv2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r move2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root move3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv3
                  COMMAND ${TOOLS_PREFIX}/rootmv move3.root:hpx move3.root:hpxpy
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 move3.root
                  OUTREF SimpleRootmv3.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r move3.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRootmv3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root move4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv4
                  COMMAND ${TOOLS_PREFIX}/rootmv move4.root:hpx move4.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 move4.root:*
                  OUTREF SimpleRootmv4.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv4Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r move4.root
                  DEPENDS
                  roottest-python-cmdLineUtils-SimpleRootmv4CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp -r test.root move5.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv5
                  COMMAND ${TOOLS_PREFIX}/rootmv move5.root:tof move5.root:dir
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv5PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls -1 move5.root:*
                  OUTREF SimpleRootmv5.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv5)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRootmv5Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm -r move5.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRootmv5CheckOutput)
#########################################################################

endif() # these tests are not run with the classic build
