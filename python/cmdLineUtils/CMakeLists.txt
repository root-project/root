configure_file(test.root . COPYONLY)

############################## PATTERN TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern1
                  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py test.root
                  OUTREF SimplePattern.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern2
                  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py test.root:tof
                  OUTREF SimplePattern2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimplePattern3
                  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/testPatternToFileNameAndPathSplitList.py test.root:*
                  OUTREF SimplePattern3.ref)
#########################################################################


############################## ROOLS TESTS ##############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools1
                  COMMAND rools test.root
                  OUTREF SimpleRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools2
                  COMMAND rools test.root:*
                  OUTREF SimpleRools2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools3
                  COMMAND rools test.root:tof
                  OUTREF SimpleRools3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRools4
                  COMMAND rools test.root:tof/*
                  OUTREF SimpleRools4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools1
                  COMMAND rools -l test.root
                  OUTREF LongRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools2
                  COMMAND rools -l test.root:*
                  OUTREF LongRools2.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools3
                  COMMAND rools -l test.root:tof
                  OUTREF LongRools3.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-LongRools4
                  COMMAND rools -l test.root:tof/*
                  OUTREF LongRools4.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRools1
                  COMMAND rools http://root.cern.ch/files/pippa.root
                  OUTREF WebRools.ref)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-WebRools2
                  COMMAND rools -l http://root.cern.ch/files/pippa.root
                  OUTREF WebRools2.ref)
#########################################################################


############################## ROORM TESTS ##############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1PrepareInput
                  COMMAND roocp test.root victim1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1
                  COMMAND roorm victim1.root:hpx
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1CheckOutput
                  COMMAND rools victim1.root
                  OUTREF SimpleRoorm.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm1Clean
                  COMMAND roorm victim1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput
                  COMMAND roocp test.root victim2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2
                  COMMAND roorm victim2.root:tof/plane0
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput
                  COMMAND rools victim2.root:tof
                  OUTREF SimpleRoorm2.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoorm2Clean
                  COMMAND roorm victim2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoorm2CheckOutput)

#########################################################################

############################# ROOMKDIR TESTS ############################
ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput
                  COMMAND roocp test.root target1.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1
                  COMMAND roomkdir target1.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput
                  COMMAND rools target1.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir1Clean
                  COMMAND roorm target1.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir1CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput
                  COMMAND roocp test.root target2.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2
                  COMMAND roomkdir target2.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput
                  COMMAND rools target2.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir2Clean
                  COMMAND roorm target2.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir2CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput
                  COMMAND roocp test.root target3.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3
                  COMMAND roomkdir target3.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput
                  COMMAND rools target3.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir3Clean
                  COMMAND roorm target3.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir3CheckOutput)



ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4PrepareInput
                  COMMAND roocp test.root target4.root)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4
                  COMMAND roomkdir target4.root:new_directory
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4PrepareInput)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4CheckOutput
                  COMMAND rools target4.root
                  OUTREF SimpleRoomkdir.ref
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4)

ROOTTEST_ADD_TEST(roottest-python-cmdLineUtils-SimpleRoomkdir4Clean
                  COMMAND roorm target4.root
                  DEPENDS roottest-python-cmdLineUtils-SimpleRoomkdir4CheckOutput)
#########################################################################
