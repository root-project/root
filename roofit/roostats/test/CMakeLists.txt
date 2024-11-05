ROOT_ADD_GTEST(testHypoTestInvResult testHypoTestInvResult.cxx
  LIBRARIES RooStats
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testHypoTestInvResult_1.root)
ROOT_ADD_GTEST(testSPlot testSPlot.cxx LIBRARIES RooStats)

#--stressRooStats----------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressRooStats stressRooStats.cxx LIBRARIES RooStats Gpad Net)
if(mathmore)
    target_compile_definitions(stressRooStats PRIVATE ROOFITMORE)
endif()

configure_file(stressRooStats_ref.root stressRooStats_ref.root COPYONLY)
if(roofit_legacy_eval_backend)
  ROOT_ADD_TEST(test-stressroostats-legacy COMMAND stressRooStats -b legacy FAILREGEX "FAILED|Error in" LABELS longtest)
endif()
ROOT_ADD_TEST(test-stressroostats-cpu COMMAND stressRooStats -b cpu FAILREGEX "FAILED|Error in" LABELS longtest)
if(cuda)
  ROOT_ADD_TEST(test-stressroostats-cuda COMMAND stressRooStats -b cuda FAILREGEX "FAILED|Error in" LABELS longtest RESOURCE_LOCK GPU)
endif()
if(roofit_legacy_eval_backend)
  ROOT_ADD_TEST(test-stressroostats-legacy-minuit2 COMMAND stressRooStats -minim Minuit2 -b legacy FAILREGEX "FAILED|Error in" LABELS longtest)
endif()
ROOT_ADD_TEST(test-stressroostats-cpu-minuit2 COMMAND stressRooStats -minim Minuit2 -b cpu FAILREGEX "FAILED|Error in" LABELS longtest)
