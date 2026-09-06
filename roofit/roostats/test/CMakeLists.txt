ROOT_ADD_GTEST(testAsymptoticCalculator testAsymptoticCalculator.cxx LIBRARIES RooStats ROOT::TestSupport)
ROOT_ADD_GTEST(testBayesianCalculator testBayesianCalculator.cxx LIBRARIES RooStats)
ROOT_ADD_GTEST(testHypoTestInvResult testHypoTestInvResult.cxx
  LIBRARIES RooStats
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testHypoTestInvResult_1.root)
ROOT_ADD_GTEST(testSPlot testSPlot.cxx LIBRARIES RooStats)

#--stressRooStats----------------------------------------------------------------------------------
# Googletest version of the old RooStats S.T.R.E.S.S. suite, parameterized
# over the RooFit evaluation backends. Like in the original suite, the tests
# are also run with the Minuit2 minimizer (the default minimizer of the suite
# is Minuit2).
ROOT_ADD_GTEST(stressRooStats stressRooStats.cxx LIBRARIES RooStats LABELS longtest TIMEOUT 3600)
if(mathmore)
    target_compile_definitions(stressRooStats PRIVATE ROOFITMORE)
endif()
if(cuda)
  set_tests_properties(gtest-roofit-roostats-stressRooStats PROPERTIES RESOURCE_LOCK GPU)
endif()
# Like the non-default-minimizer variant in the original suite, the Minuit
# variant only runs the legacy and cpu backends (that is also why it doesn't
# need to lock the GPU resource).
ROOT_ADD_TEST(test-stressroostats-minuit COMMAND stressRooStats --gtest_filter=-*EvalBackendcuda*
              ENVIRONMENT STRESSROOSTATS_MINIMIZER=Minuit LABELS longtest TIMEOUT 3600)
