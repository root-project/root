if (ROOT_pyroot_FOUND AND NOT ROOT_CLASSIC_BUILD) # Do not run with classic build

configure_file(test.root . COPYONLY)
configure_file(simpleTree.root . COPYONLY)
configure_file(multipleNameCycles.root . COPYONLY)
configure_file(MakeNameCyclesRootmvInput.C . COPYONLY)

############################## PATTERN TESTS ############################
set (TESTPATTERN_EXE ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py)
ROOTTEST_ADD_TEST(SimplePattern1
                  COMMAND ${TESTPATTERN_EXE} test.root
                  OUTREF SimplePattern.ref)

ROOTTEST_ADD_TEST(SimplePattern2
                  COMMAND ${TESTPATTERN_EXE} test.root:tof
                  OUTREF SimplePattern2.ref)

ROOTTEST_ADD_TEST(SimplePattern3
                  COMMAND ${TESTPATTERN_EXE} test.root:*
                  OUTREF SimplePattern3.ref)
#########################################################################

if(MSVC)
    # the command line tools works fine in the Windows command prompt but
    # not from CTest, so let's add Python.exe and the .py file extension
    set(TOOLS_PREFIX ${PYTHON_EXECUTABLE} ${ROOTSYS}/bin)
    set(pyext .py)
else()
    set(TOOLS_PREFIX ${ROOTSYS}/bin)
endif()

############################## ROOLS TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 test.root
                  OUTREF SimpleRootls.ref)

ROOTTEST_ADD_TEST(SimpleRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 test.root:*
                  OUTREF SimpleRootls2.ref)

ROOTTEST_ADD_TEST(SimpleRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 test.root:tof
                  OUTREF SimpleRootls3.ref)

ROOTTEST_ADD_TEST(SimpleRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 test.root:tof/*
                  OUTREF SimpleRootls4.ref)

ROOTTEST_ADD_TEST(LongRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -l test.root
                  OUTREF LongRootls.ref)

ROOTTEST_ADD_TEST(LongRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -l test.root:*
                  OUTREF LongRootls2.ref)

ROOTTEST_ADD_TEST(LongRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -l test.root:tof
                  OUTREF LongRootls3.ref)

ROOTTEST_ADD_TEST(LongRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -l test.root:tof/*
                  OUTREF LongRootls4.ref)

ROOTTEST_ADD_TEST(WebRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 http://root.cern.ch/files/pippa.root
                  OUTREF WebRootls.ref)

ROOTTEST_ADD_TEST(WebRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 http://root.cern.ch/files/pippa.root:LL
                  OUTREF WebRootls2.ref)

ROOTTEST_ADD_TEST(TreeRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -t simpleTree.root
                  OUTREF TreeRootls.ref)

ROOTTEST_ADD_TEST(NameCyclesRootls
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -t multipleNameCycles.root:tree
                  OUTREF MultipleNameCyclesRootls.ref)

#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootrm1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim1.root)

ROOTTEST_ADD_TEST(SimpleRootrm1
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} victim1.root:hpx
                  DEPENDS SimpleRootrm1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 victim1.root
                  OUTREF SimpleRootrm.ref
                  DEPENDS SimpleRootrm1)

ROOTTEST_ADD_TEST(SimpleRootrm1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r victim1.root
                  DEPENDS SimpleRootrm1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootrm2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim2.root)

ROOTTEST_ADD_TEST(SimpleRootrm2
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} victim2.root:hp*
                  DEPENDS SimpleRootrm2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 victim2.root
                  OUTREF SimpleRootrm2.ref
                  DEPENDS SimpleRootrm2)

ROOTTEST_ADD_TEST(SimpleRootrm2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r victim2.root
                  DEPENDS SimpleRootrm2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootrm3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim3.root)

ROOTTEST_ADD_TEST(SimpleRootrm3
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r victim3.root:tof/plane0
                  DEPENDS SimpleRootrm3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 victim3.root:tof
                  OUTREF SimpleRootrm3.ref
                  DEPENDS SimpleRootrm3)

ROOTTEST_ADD_TEST(SimpleRootrm3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r victim3.root
                  DEPENDS SimpleRootrm3CheckOutput)
#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmkdir1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target1.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir1
                  COMMAND ${TOOLS_PREFIX}/rootmkdir${pyext} target1.root:new_directory
                  DEPENDS SimpleRootmkdir1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 target1.root
                  OUTREF SimpleRootmkdir.ref
                  DEPENDS SimpleRootmkdir1)

ROOTTEST_ADD_TEST(SimpleRootmkdir1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r target1.root
                  DEPENDS SimpleRootmkdir1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmkdir2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target2.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir2
                  COMMAND ${TOOLS_PREFIX}/rootmkdir${pyext} target2.root:dir1 target2.root:dir2
                  DEPENDS SimpleRootmkdir2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 target2.root
                  OUTREF SimpleRootmkdir2.ref
                  DEPENDS SimpleRootmkdir2)

ROOTTEST_ADD_TEST(SimpleRootmkdir2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r target2.root
                  DEPENDS SimpleRootmkdir2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmkdir3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target3.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir3
                  COMMAND ${TOOLS_PREFIX}/rootmkdir${pyext} -p target3.root:aa/bb/cc
                  DEPENDS SimpleRootmkdir3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 target3.root:aa/bb
                  OUTREF SimpleRootmkdir3.ref
                  DEPENDS SimpleRootmkdir3)

ROOTTEST_ADD_TEST(SimpleRootmkdir3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r target3.root
                  DEPENDS SimpleRootmkdir3CheckOutput)
#########################################################################

############################# ROOCP TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootcp1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy1.root)

ROOTTEST_ADD_TEST(SimpleRootcp1
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} copy1.root:hpx copy1.root:histo
                  DEPENDS SimpleRootcp1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 copy1.root
                  OUTREF SimpleRootcp.ref
                  DEPENDS SimpleRootcp1)

ROOTTEST_ADD_TEST(SimpleRootcp1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r copy1.root
                  DEPENDS SimpleRootcp1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy2.root)

ROOTTEST_ADD_TEST(SimpleRootcp2
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} -r copy2.root:tof copy2.root:fot
                  DEPENDS SimpleRootcp2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 copy2.root
                  OUTREF SimpleRootcp2.ref
                  DEPENDS SimpleRootcp2)

ROOTTEST_ADD_TEST(SimpleRootcp2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r copy2.root
                  DEPENDS SimpleRootcp2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy3.root)

ROOTTEST_ADD_TEST(SimpleRootcp3
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --replace copy3.root:hpx copy3.root:hpxpy
                  DEPENDS SimpleRootcp3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 copy3.root
                  OUTREF SimpleRootcp3.ref
                  DEPENDS SimpleRootcp3)

ROOTTEST_ADD_TEST(SimpleRootcp3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r copy3.root
                  DEPENDS SimpleRootcp3CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootcp4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy4.root)

ROOTTEST_ADD_TEST(SimpleRootcp4
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} copy4.root:hpx copy4.root:dir
                  DEPENDS SimpleRootcp4PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 copy4.root:dir
                  OUTREF SimpleRootcp4.ref
                  DEPENDS SimpleRootcp4)

ROOTTEST_ADD_TEST(SimpleRootcp4Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r copy4.root
                  DEPENDS SimpleRootcp4CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy5.root)

ROOTTEST_ADD_TEST(SimpleRootcp5
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} -r copy5.root:tof copy5.root:dir
                  DEPENDS SimpleRootcp5PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 copy5.root:dir
                  OUTREF SimpleRootcp5.ref
                  DEPENDS SimpleRootcp5)

ROOTTEST_ADD_TEST(SimpleRootcp5Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r copy5.root
                  DEPENDS SimpleRootcp5CheckOutput)
#########################################################################

############################# ROOMV TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmv1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move1.root)

ROOTTEST_ADD_TEST(SimpleRootmv1
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} move1.root:hpx move1.root:histo
                  DEPENDS SimpleRootmv1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 move1.root
                  OUTREF SimpleRootmv.ref
                  DEPENDS SimpleRootmv1)

ROOTTEST_ADD_TEST(SimpleRootmv1Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r move1.root
                  DEPENDS SimpleRootmv1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmv2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move2.root)

ROOTTEST_ADD_TEST(SimpleRootmv2
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} move2.root:tof move2.root:fot
                  DEPENDS SimpleRootmv2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 move2.root
                  OUTREF SimpleRootmv2.ref
                  DEPENDS SimpleRootmv2)

ROOTTEST_ADD_TEST(SimpleRootmv2Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r move2.root
                  DEPENDS SimpleRootmv2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmv3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move3.root)

ROOTTEST_ADD_TEST(SimpleRootmv3
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} move3.root:hpx move3.root:hpxpy
                  DEPENDS SimpleRootmv3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 move3.root
                  OUTREF SimpleRootmv3.ref
                  DEPENDS SimpleRootmv3)

ROOTTEST_ADD_TEST(SimpleRootmv3Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r move3.root
                  DEPENDS SimpleRootmv3CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootmv4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move4.root)

ROOTTEST_ADD_TEST(SimpleRootmv4
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} move4.root:hpx move4.root:dir
                  DEPENDS SimpleRootmv4PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 move4.root:*
                  OUTREF SimpleRootmv4.ref
                  DEPENDS SimpleRootmv4)

ROOTTEST_ADD_TEST(SimpleRootmv4Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r move4.root
                  DEPENDS SimpleRootmv4CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootmv5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move5.root)

ROOTTEST_ADD_TEST(SimpleRootmv5
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} move5.root:tof move5.root:dir
                  DEPENDS SimpleRootmv5PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 move5.root:*
                  OUTREF SimpleRootmv5.ref
                  DEPENDS SimpleRootmv5)

ROOTTEST_ADD_TEST(SimpleRootmv5Clean
                  COMMAND ${TOOLS_PREFIX}/rootrm${pyext} -r move5.root
                  DEPENDS SimpleRootmv5CheckOutput)


ROOTTEST_ADD_TEST(NameCyclesRootmvPrepareInput
                  COMMAND root.exe -b -q -l MakeNameCyclesRootmvInput.C)

ROOTTEST_ADD_TEST(NameCyclesRootmv
                  COMMAND ${TOOLS_PREFIX}/rootmv${pyext} nameCyclesRootmvInput.root:tree nameCyclesRootmvOutput.root
                  DEPENDS NameCyclesRootmvPrepareInput)

ROOTTEST_ADD_TEST(NameCyclesRootmvCheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${pyext} -1 nameCyclesRootmvOutput.root
                  OUTREF NameCyclesRootmvCheckOutput.ref
                  DEPENDS NameCyclesRootmv)
#########################################################################

ROOTTEST_ADD_TEST(ROOT_8197
                  MACRO ROOT_8197.C
                  OUTREF ROOT_8197.ref)

endif() # these tests are not run with the classic build
