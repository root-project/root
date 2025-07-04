# For now let's reuse the rootls.py test files.
set(ROOTFILES_DIR ../python/cmdLineUtils)

configure_file(${ROOTFILES_DIR}/test.root . COPYONLY)
configure_file(${ROOTFILES_DIR}/simpleTree.root . COPYONLY)
configure_file(${ROOTFILES_DIR}/multipleNameCycles.root . COPYONLY)
configure_file(${ROOTFILES_DIR}/MakeNameCyclesRootmvInput.C . COPYONLY)
configure_file(${ROOTFILES_DIR}/subdirs.root . COPYONLY)

# We compare the output of some tests against ref. files, and we don't want to
# get noise for Python warnings (for example the warning that the GIL is
# activated because the ROOT Python bindings don't support free threading yet).
set(test_env PYTHONWARNINGS=ignore)

if(MSVC)
    # the command line tools works fine in the Windows command prompt but
    # not from CTest, so let's add Python.exe and the .py file extension
    set(PY_TOOLS_PREFIX ${Python3_EXECUTABLE} ${ROOTSYS}/bin)
    set(TOOLS_PREFIX ${ROOTSYS}/bin)
    set(pyext .py)
else()
    set(PY_TOOLS_PREFIX ${ROOTSYS}/bin)
    set(TOOLS_PREFIX ${ROOTSYS}/bin)
endif()

############################## ROOLS TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootlsCxx1
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 test.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootls.ref)

ROOTTEST_ADD_TEST(SimpleRootlsCxx2
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 test.root:*
                  OUTREF ${ROOTFILES_DIR}/SimpleRootls2.ref)

ROOTTEST_ADD_TEST(SimpleRootlsCxx3
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 test.root:tof
                  OUTREF ${ROOTFILES_DIR}/SimpleRootls3.ref)

ROOTTEST_ADD_TEST(SimpleRootlsCxx4
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 test.root:tof/*
                  OUTREF ${ROOTFILES_DIR}/SimpleRootls4.ref)

ROOTTEST_ADD_TEST(LongRootlsCxx1
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -l test.root
                  OUTREF ${ROOTFILES_DIR}/LongRootls.ref)

ROOTTEST_ADD_TEST(LongRootlsCxx2
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -l test.root:*
                  OUTREF ${ROOTFILES_DIR}/LongRootls2.ref)

ROOTTEST_ADD_TEST(LongRootlsCxx3
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -l test.root:tof
                  OUTREF ${ROOTFILES_DIR}/LongRootls3.ref)

ROOTTEST_ADD_TEST(LongRootlsCxx4
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -l test.root:tof/*
                  OUTREF ${ROOTFILES_DIR}/LongRootls4.ref)

if(davix AND NOT MSVC) # davix is optional ; Windows cannot yet read https
    ROOTTEST_ADD_TEST(WebRootlsCxx1
                    COMMAND ${TOOLS_PREFIX}/rootlscxx -1 http://root.cern.ch/files/pippa.root
                    OUTREF ${ROOTFILES_DIR}/WebRootls.ref)

    ROOTTEST_ADD_TEST(WebRootlsCxx2
                    COMMAND ${TOOLS_PREFIX}/rootlscxx -1 http://root.cern.ch/files/pippa.root:LL
                    OUTREF ${ROOTFILES_DIR}/WebRootls2.ref)
endif()

ROOTTEST_ADD_TEST(TreeRootlsCxx1
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -t simpleTree.root
                  OUTREF ${ROOTFILES_DIR}/TreeRootls.ref)

ROOTTEST_ADD_TEST(NameCyclesRootlsCxx
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -t multipleNameCycles.root:tree
                  OUTREF ${ROOTFILES_DIR}/MultipleNameCyclesRootls.ref)

ROOTTEST_ADD_TEST(RecursiveRootlsCxx
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -lr subdirs.root
                  OUTREF ${ROOTFILES_DIR}/RecursiveRootls.ref)

#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(SimpleRootrm1PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim1.root)

ROOTTEST_ADD_TEST(SimpleRootrm1
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} victim1.root:hpx
                  DEPENDS SimpleRootrm1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 victim1.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootrm.ref
                  DEPENDS SimpleRootrm1
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim1.root
                  DEPENDS SimpleRootrm1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootrm2PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim2.root)

ROOTTEST_ADD_TEST(SimpleRootrm2
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} victim2.root:hp*
                  DEPENDS SimpleRootrm2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 victim2.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootrm2.ref
                  DEPENDS SimpleRootrm2
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim2.root
                  DEPENDS SimpleRootrm2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootrm3PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root victim3.root)

ROOTTEST_ADD_TEST(SimpleRootrm3
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim3.root:tof/plane0
                  DEPENDS SimpleRootrm3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootrm3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 victim3.root:tof
                  OUTREF ${ROOTFILES_DIR}/SimpleRootrm3.ref
                  DEPENDS SimpleRootrm3
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootrm3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r victim3.root
                  DEPENDS SimpleRootrm3CheckOutput)
#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmkdir1PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target1.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir1
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} target1.root:new_directory
                  DEPENDS SimpleRootmkdir1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 target1.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmkdir.ref
                  DEPENDS SimpleRootmkdir1
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target1.root
                  DEPENDS SimpleRootmkdir1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmkdir2PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target2.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir2
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} target2.root:dir1 target2.root:dir2
                  DEPENDS SimpleRootmkdir2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 target2.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmkdir2.ref
                  DEPENDS SimpleRootmkdir2
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target2.root
                  DEPENDS SimpleRootmkdir2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmkdir3PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root target3.root)

ROOTTEST_ADD_TEST(SimpleRootmkdir3
                  COMMAND ${PY_TOOLS_PREFIX}/rootmkdir${pyext} -p target3.root:aa/bb/cc
                  DEPENDS SimpleRootmkdir3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmkdir3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 target3.root:aa/bb
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmkdir3.ref
                  DEPENDS SimpleRootmkdir3
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmkdir3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r target3.root
                  DEPENDS SimpleRootmkdir3CheckOutput)
#########################################################################

############################# ROOCP TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootcp1PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy1.root)

ROOTTEST_ADD_TEST(SimpleRootcp1
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} copy1.root:hpx copy1.root:histo
                  DEPENDS SimpleRootcp1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 copy1.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootcp.ref
                  DEPENDS SimpleRootcp1
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy1.root
                  DEPENDS SimpleRootcp1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp2PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy2.root)

ROOTTEST_ADD_TEST(SimpleRootcp2
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} -r copy2.root:tof copy2.root:fot
                  DEPENDS SimpleRootcp2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 copy2.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootcp2.ref
                  DEPENDS SimpleRootcp2
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy2.root
                  DEPENDS SimpleRootcp2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp3PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy3.root)

ROOTTEST_ADD_TEST(SimpleRootcp3
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --replace copy3.root:hpx copy3.root:hpxpy
                  DEPENDS SimpleRootcp3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 copy3.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootcp3.ref
                  DEPENDS SimpleRootcp3
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy3.root
                  DEPENDS SimpleRootcp3CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootcp4PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy4.root)

ROOTTEST_ADD_TEST(SimpleRootcp4
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} copy4.root:hpx copy4.root:dir
                  DEPENDS SimpleRootcp4PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 copy4.root:dir
                  OUTREF ${ROOTFILES_DIR}/SimpleRootcp4.ref
                  DEPENDS SimpleRootcp4
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp4Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy4.root
                  DEPENDS SimpleRootcp4CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootcp5PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root copy5.root)

ROOTTEST_ADD_TEST(SimpleRootcp5
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} -r copy5.root:tof copy5.root:dir
                  DEPENDS SimpleRootcp5PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootcp5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 copy5.root:dir
                  OUTREF ${ROOTFILES_DIR}/SimpleRootcp5.ref
                  DEPENDS SimpleRootcp5
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootcp5Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r copy5.root
                  DEPENDS SimpleRootcp5CheckOutput)
#########################################################################

############################# ROOMV TESTS ############################
ROOTTEST_ADD_TEST(SimpleRootmv1PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move1.root)

ROOTTEST_ADD_TEST(SimpleRootmv1
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move1.root:hpx move1.root:histo
                  DEPENDS SimpleRootmv1PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv1CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 move1.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmv.ref
                  DEPENDS SimpleRootmv1
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv1Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move1.root
                  DEPENDS SimpleRootmv1CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmv2PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move2.root)

ROOTTEST_ADD_TEST(SimpleRootmv2
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move2.root:tof move2.root:fot
                  DEPENDS SimpleRootmv2PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv2CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 move2.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmv2.ref
                  DEPENDS SimpleRootmv2
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv2Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move2.root
                  DEPENDS SimpleRootmv2CheckOutput)



ROOTTEST_ADD_TEST(SimpleRootmv3PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move3.root)

ROOTTEST_ADD_TEST(SimpleRootmv3
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move3.root:hpx move3.root:hpxpy
                  DEPENDS SimpleRootmv3PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv3CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 move3.root
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmv3.ref
                  DEPENDS SimpleRootmv3
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv3Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move3.root
                  DEPENDS SimpleRootmv3CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootmv4PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move4.root)

ROOTTEST_ADD_TEST(SimpleRootmv4
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move4.root:hpx move4.root:dir
                  DEPENDS SimpleRootmv4PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv4CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 move4.root:*
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmv4.ref
                  DEPENDS SimpleRootmv4
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv4Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move4.root
                  DEPENDS SimpleRootmv4CheckOutput)

ROOTTEST_ADD_TEST(SimpleRootmv5PrepareInput
                  COMMAND ${PY_TOOLS_PREFIX}/rootcp${pyext} --recreate -r test.root move5.root)

ROOTTEST_ADD_TEST(SimpleRootmv5
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} move5.root:tof move5.root:dir
                  DEPENDS SimpleRootmv5PrepareInput)

ROOTTEST_ADD_TEST(SimpleRootmv5CheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 move5.root:*
                  OUTREF ${ROOTFILES_DIR}/SimpleRootmv5.ref
                  DEPENDS SimpleRootmv5
                  ENVIRONMENT ${test_env})

ROOTTEST_ADD_TEST(SimpleRootmv5Clean
                  COMMAND ${PY_TOOLS_PREFIX}/rootrm${pyext} -r move5.root
                  DEPENDS SimpleRootmv5CheckOutput)


ROOTTEST_ADD_TEST(NameCyclesRootmvPrepareInput
                  COMMAND root.exe -b -q -l MakeNameCyclesRootmvInput.C)

ROOTTEST_ADD_TEST(NameCyclesRootmv
                  COMMAND ${PY_TOOLS_PREFIX}/rootmv${pyext} nameCyclesRootmvInput.root:tree nameCyclesRootmvOutput.root
                  DEPENDS NameCyclesRootmvPrepareInput)

ROOTTEST_ADD_TEST(NameCyclesRootmvCheckOutput
                  COMMAND ${TOOLS_PREFIX}/rootlscxx -1 nameCyclesRootmvOutput.root
                  OUTREF ${ROOTFILES_DIR}/NameCyclesRootmvCheckOutput.ref
                  DEPENDS NameCyclesRootmv
                  ENVIRONMENT ${test_env})
#########################################################################
