configure_file(test.root . COPYONLY)
configure_file(simpleTree.root . COPYONLY)
configure_file(multipleNameCycles.root . COPYONLY)
configure_file(MakeNameCyclesRootmvInput.C . COPYONLY)
configure_file(subdirs.root . COPYONLY)
configure_file(nested.root . COPYONLY)
configure_file(RNTuple.root . COPYONLY)

# We should explicitly use the Python executable from the Python version that
# was used to build ROOT. Otherwise, we risk picking up a different Python
# version at test time.
set(PY_TOOLS_PREFIX ${Python3_EXECUTABLE} ${ROOTSYS}/bin)
set(TOOLS_PREFIX ${ROOTSYS}/bin)

if(MSVC)
    set(pyext .py)
    set(exeext .exe)
endif()

############################## ROOTLS TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 test.root
                  OUTREF SimpleRootls.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 test.root:*
                  OUTREF SimpleRootls2.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 test.root:tof
                  OUTREF SimpleRootls3.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 test.root:tof/*
                  OUTREF SimpleRootls4.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootls5
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1c test.root
                  OUTREF SimpleRootls5.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(LongRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -l test.root
                  OUTREF LongRootls.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(LongRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -l test.root:*
                  OUTREF LongRootls2.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(LongRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -l test.root:tof
                  OUTREF LongRootls3.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(LongRootls4
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -l test.root:tof/*
                  OUTREF LongRootls4.ref
                  ENVIRONMENT ${test_env})

if(davix AND NOT MSVC) # davix is optional ; Windows cannot yet read https
    ROOTTEST_ADD_TEST(WebRootls1
                    COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 http://root.cern.ch/files/pippa.root
                    OUTREF WebRootls.ref
                    ENVIRONMENT ${test_env})

    ROOTTEST_ADD_TEST(WebRootls2
                    COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 http://root.cern.ch/files/pippa.root:LL
                    OUTREF WebRootls2.ref
                    ENVIRONMENT ${test_env})
endif()

ROOTTEST_ADD_TEST(TreeRootls1
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -t simpleTree.root
                  OUTREF TreeRootls.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(NameCyclesRootls
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -t multipleNameCycles.root:tree
                  OUTREF MultipleNameCyclesRootls.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(NameCyclesRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -c1 multipleNameCycles.root:tree
                  OUTREF MultipleNameCyclesRootls2.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RecursiveRootls
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -lr subdirs.root
                  OUTREF RecursiveRootls.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RecursiveRootls2
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -lr nested.root
                  OUTREF RecursiveRootls2.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RecursiveRootls3
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -r subdirs.root
                  OUTREF RecursiveRootls3.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RootlsRNTuple
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -R RNTuple.root
                  OUTREF RootlsRNTuple.ref
                  ENVIRONMENT ${test_env})

#########################################################################

if(pyroot)

ROOTTEST_ADD_TEST(ROOT_8197
                  MACRO ROOT_8197.C
                  OUTREF ROOT_8197.ref
                  ENVIRONMENT ${test_env})

# We compare the output of some tests against ref. files, and we don't want to
# get noise for Python warnings (for example the warning that the GIL is
# activated because the ROOT Python bindings don't support free threading yet).
set(test_env PYTHONWARNINGS=ignore)

############################## PATTERN TESTS ############################
set (TESTPATTERN_EXE ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py)
ROOTTEST_ADD_TEST(SimplePattern1
                  COMMAND ${TESTPATTERN_EXE} test.root
                  OUTREF SimplePattern.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimplePattern2
                  COMMAND ${TESTPATTERN_EXE} test.root:tof
                  OUTREF SimplePattern2.ref
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimplePattern3
                  COMMAND ${TESTPATTERN_EXE} test.root:*
                  OUTREF SimplePattern3.ref
                  ENVIRONMENT ${test_env})
#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootrm1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root victim1.root
                  FIXTURES_SETUP main-SimpleRootrm1PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm1
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} victim1.root:hpx
                  FIXTURES_REQUIRED main-SimpleRootrm1PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootrm1-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 victim1.root
                  OUTREF SimpleRootrm.ref
                  FIXTURES_REQUIRED main-SimpleRootrm1-fixture
                  FIXTURES_SETUP main-SimpleRootrm1CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim1.root
                  FIXTURES_REQUIRED main-SimpleRootrm1CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootrm2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root victim2.root
                  FIXTURES_SETUP main-SimpleRootrm2PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm2
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} victim2.root:hp*
                  FIXTURES_REQUIRED main-SimpleRootrm2PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootrm2-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 victim2.root
                  OUTREF SimpleRootrm2.ref
                  FIXTURES_REQUIRED main-SimpleRootrm2-fixture
                  FIXTURES_SETUP main-SimpleRootrm2CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim2.root
                  FIXTURES_REQUIRED main-SimpleRootrm2CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootrm3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root victim3.root
                  FIXTURES_SETUP main-SimpleRootrm3PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm3
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim3.root:tof/plane0
                  FIXTURES_REQUIRED main-SimpleRootrm3PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootrm3-fixture)

ROOTTEST_ADD_TEST(SimpleRootrm3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 victim3.root:tof
                  OUTREF SimpleRootrm3.ref
                  FIXTURES_REQUIRED main-SimpleRootrm3-fixture
                  FIXTURES_SETUP main-SimpleRootrm3CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim3.root
                  FIXTURES_REQUIRED main-SimpleRootrm3CheckOutput-fixture)
#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmkdir1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root target1.root
                  FIXTURES_SETUP main-SimpleRootmkdir1PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir1
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} target1.root:new_directory
                  FIXTURES_REQUIRED main-SimpleRootmkdir1PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir1-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 target1.root
                  OUTREF SimpleRootmkdir.ref
                  FIXTURES_REQUIRED main-SimpleRootmkdir1-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir1CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target1.root
                  FIXTURES_REQUIRED main-SimpleRootmkdir1CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootmkdir2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root target2.root
                  FIXTURES_SETUP main-SimpleRootmkdir2PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir2
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} target2.root:dir1 target2.root:dir2
                  FIXTURES_REQUIRED main-SimpleRootmkdir2PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir2-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 target2.root
                  OUTREF SimpleRootmkdir2.ref
                  FIXTURES_REQUIRED main-SimpleRootmkdir2-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir2CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target2.root
                  FIXTURES_REQUIRED main-SimpleRootmkdir2CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootmkdir3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root target3.root
                  FIXTURES_SETUP main-SimpleRootmkdir3PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir3
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} -p target3.root:aa/bb/cc
                  FIXTURES_REQUIRED main-SimpleRootmkdir3PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir3-fixture)

ROOTTEST_ADD_TEST(SimpleRootmkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 target3.root:aa/bb
                  OUTREF SimpleRootmkdir3.ref
                  FIXTURES_REQUIRED main-SimpleRootmkdir3-fixture
                  FIXTURES_SETUP main-SimpleRootmkdir3CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target3.root
                  FIXTURES_REQUIRED main-SimpleRootmkdir3CheckOutput-fixture)
#########################################################################

############################# ROOTCP TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootcp1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy1.root
                  FIXTURES_SETUP main-SimpleRootcp1PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp1
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} copy1.root:hpx copy1.root:histo
                  FIXTURES_REQUIRED main-SimpleRootcp1PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootcp1-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy1.root
                  OUTREF SimpleRootcp.ref
                  FIXTURES_REQUIRED main-SimpleRootcp1-fixture
                  FIXTURES_SETUP main-SimpleRootcp1CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy1.root
                  FIXTURES_REQUIRED main-SimpleRootcp1CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootcp2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy2.root
                  FIXTURES_SETUP main-SimpleRootcp2PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp2
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} -r copy2.root:tof copy2.root:fot
                  FIXTURES_REQUIRED main-SimpleRootcp2PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootcp2-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy2.root
                  OUTREF SimpleRootcp2.ref
                  FIXTURES_REQUIRED main-SimpleRootcp2-fixture
                  FIXTURES_SETUP main-SimpleRootcp2CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy2.root
                  FIXTURES_REQUIRED main-SimpleRootcp2CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootcp3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy3.root
                  FIXTURES_SETUP main-SimpleRootcp3PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp3
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --replace copy3.root:hpx copy3.root:hpxpy
                  FIXTURES_REQUIRED main-SimpleRootcp3PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootcp3-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy3.root
                  OUTREF SimpleRootcp3.ref
                  FIXTURES_REQUIRED main-SimpleRootcp3-fixture
                  FIXTURES_SETUP main-SimpleRootcp3CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy3.root
                  FIXTURES_REQUIRED main-SimpleRootcp3CheckOutput-fixture)


ROOTTEST_ADD_TEST(SimpleRootcp4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy4.root
                  FIXTURES_SETUP main-SimpleRootcp4PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp4
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} copy4.root:hpx copy4.root:dir
                  FIXTURES_REQUIRED main-SimpleRootcp4PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootcp4-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy4.root:dir
                  OUTREF SimpleRootcp4.ref
                  FIXTURES_REQUIRED main-SimpleRootcp4-fixture
                  FIXTURES_SETUP main-SimpleRootcp4CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp4Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy4.root
                  FIXTURES_REQUIRED main-SimpleRootcp4CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootcp5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy5.root
                  FIXTURES_SETUP main-SimpleRootcp5PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp5
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} -r copy5.root:tof copy5.root:dir
                  FIXTURES_REQUIRED main-SimpleRootcp5PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootcp5-fixture)

ROOTTEST_ADD_TEST(SimpleRootcp5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy5.root:dir
                  OUTREF SimpleRootcp5.ref
                  FIXTURES_REQUIRED main-SimpleRootcp5-fixture
                  FIXTURES_SETUP main-SimpleRootcp5CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp5Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy5.root
                  FIXTURES_REQUIRED main-SimpleRootcp5CheckOutput-fixture)

ROOTTEST_ADD_TEST(RootcpReplaceRecursivePrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root copy_replace_recursive.root
                  FIXTURES_SETUP main-RootcpReplaceRecursivePrepareInput-fixture)

ROOTTEST_ADD_TEST(RootcpReplaceRecursiveWrong
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --replace -r copy_replace_recursive.root:tof copy_replace_recursive.root
                  FIXTURES_REQUIRED main-RootcpReplaceRecursivePrepareInput-fixture
                  FIXTURES_SETUP main-RootcpReplaceRecursiveWrong-fixture
                  PASSREGEX "Error in <rootcp>: copy_replace_recursive.root:tof: source and destination cannot be the same")

ROOTTEST_ADD_TEST(RootcpReplaceRecursive
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --replace -r test.root:tof copy_replace_recursive.root
                  FIXTURES_REQUIRED main-RootcpReplaceRecursiveWrong-fixture
                  FIXTURES_SETUP main-RootcpReplaceRecursive-fixture)

ROOTTEST_ADD_TEST(RootcpReplaceRecursiveCheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 copy_replace_recursive.root:tof
                  OUTREF RootcpReplaceRecursive.ref
                  FIXTURES_REQUIRED main-RootcpReplaceRecursive-fixture
                  FIXTURES_SETUP main-RootcpReplaceRecursiveCheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RootcpReplaceRecursiveClean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy_replace_recursive.root
                  FIXTURES_REQUIRED main-RootcpReplaceRecursiveCheckOutput-fixture)


ROOTTEST_ADD_TEST(SimpleRootcpInexistent
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} test.root:does_not_exist test.root:
                  PASSREGEX "Error in <rootcp>: 'test.root:does_not_exist' matches no objects.")

if(davix)
ROOTTEST_ADD_TEST(RootcpReadRemote
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root copy_remote.root
                  FIXTURES_SETUP main-RootcpReadRemote-fixture)

ROOTTEST_ADD_TEST(RootcpReadRemoteCheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} copy_remote.root
                  OUTREF RootcpReadRemote.ref
                  FIXTURES_REQUIRED main-RootcpReadRemote-fixture
                  FIXTURES_SETUP main-RootcpReadRemoteCheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(RootcpReadRemoteClean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy_remote.root
                  FIXTURES_REQUIRED main-RootcpReadRemoteCheckOutput-fixture)
endif()

#########################################################################

############################# ROOMV TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmv1PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root move1.root
                  FIXTURES_SETUP main-SimpleRootmv1PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv1
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move1.root:hpx move1.root:histo
                  FIXTURES_REQUIRED main-SimpleRootmv1PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmv1-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 move1.root
                  OUTREF SimpleRootmv.ref
                  FIXTURES_REQUIRED main-SimpleRootmv1-fixture
                  FIXTURES_SETUP main-SimpleRootmv1CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move1.root
                  FIXTURES_REQUIRED main-SimpleRootmv1CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootmv2PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root move2.root
                  FIXTURES_SETUP main-SimpleRootmv2PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv2
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move2.root:tof move2.root:fot
                  FIXTURES_REQUIRED main-SimpleRootmv2PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmv2-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 move2.root
                  OUTREF SimpleRootmv2.ref
                  FIXTURES_REQUIRED main-SimpleRootmv2-fixture
                  FIXTURES_SETUP main-SimpleRootmv2CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move2.root
                  FIXTURES_REQUIRED main-SimpleRootmv2CheckOutput-fixture)



ROOTTEST_ADD_TEST(SimpleRootmv3PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root move3.root
                  FIXTURES_SETUP main-SimpleRootmv3PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv3
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move3.root:hpx move3.root:hpxpy
                  FIXTURES_REQUIRED main-SimpleRootmv3PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmv3-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 move3.root
                  OUTREF SimpleRootmv3.ref
                  FIXTURES_REQUIRED main-SimpleRootmv3-fixture
                  FIXTURES_SETUP main-SimpleRootmv3CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move3.root
                  FIXTURES_REQUIRED main-SimpleRootmv3CheckOutput-fixture)


ROOTTEST_ADD_TEST(SimpleRootmv4PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root move4.root
                  FIXTURES_SETUP main-SimpleRootmv4PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv4
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move4.root:hpx move4.root:dir
                  FIXTURES_REQUIRED main-SimpleRootmv4PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmv4-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 move4.root:*
                  OUTREF SimpleRootmv4.ref
                  FIXTURES_REQUIRED main-SimpleRootmv4-fixture
                  FIXTURES_SETUP main-SimpleRootmv4CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv4Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move4.root
                  FIXTURES_REQUIRED main-SimpleRootmv4CheckOutput-fixture)


ROOTTEST_ADD_TEST(SimpleRootmv5PrepareInput
                  COMMAND ${TOOLS_PREFIX}/rootcp${exeext} --recreate -r test.root move5.root
                  FIXTURES_SETUP main-SimpleRootmv5PrepareInput-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv5
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move5.root:tof move5.root:dir
                  FIXTURES_REQUIRED main-SimpleRootmv5PrepareInput-fixture
                  FIXTURES_SETUP main-SimpleRootmv5-fixture)

ROOTTEST_ADD_TEST(SimpleRootmv5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 move5.root:*
                  OUTREF SimpleRootmv5.ref
                  FIXTURES_REQUIRED main-SimpleRootmv5-fixture
                  FIXTURES_SETUP main-SimpleRootmv5CheckOutput-fixture
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv5Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move5.root
                  FIXTURES_REQUIRED main-SimpleRootmv5CheckOutput-fixture)


ROOTTEST_ADD_TEST(NameCyclesRootmvPrepareInput
                  COMMAND root.exe -b -q -l MakeNameCyclesRootmvInput.C
                  FIXTURES_SETUP main-NameCyclesRootmvPrepareInput-fixture)

ROOTTEST_ADD_TEST(NameCyclesRootmv
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} nameCyclesRootmvInput.root:tree nameCyclesRootmvOutput.root
                  FIXTURES_REQUIRED main-NameCyclesRootmvPrepareInput-fixture
                  FIXTURES_SETUP main-NameCyclesRootmv-fixture)

ROOTTEST_ADD_TEST(NameCyclesRootmvCheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootls${exeext} -1 nameCyclesRootmvOutput.root
                  OUTREF NameCyclesRootmvCheckOutput.ref
                  FIXTURES_REQUIRED main-NameCyclesRootmv-fixture
                  ENVIRONMENT ${test_env})
#########################################################################

endif(pyroot)
