# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT roostats package
# @author Pere Mato, CERN
############################################################################

add_library(RooStats SHARED)
target_sources(RooStats
PRIVATE
  src/AsymptoticCalculator.cxx
  src/BayesianCalculator.cxx
  src/BernsteinCorrection.cxx
  src/ConfidenceBelt.cxx
  src/DetailedOutputAggregator.cxx
  src/FeldmanCousins.cxx
  src/FrequentistCalculator.cxx
  src/Heaviside.cxx
  src/HybridCalculator.cxx
  src/HybridResult.cxx
  src/HypoTestCalculatorGeneric.cxx
  src/HypoTestInverter.cxx
  src/HypoTestInverterPlot.cxx
  src/HypoTestInverterResult.cxx
  src/HypoTestPlot.cxx
  src/HypoTestResult.cxx
  src/LikelihoodInterval.cxx
  src/LikelihoodIntervalPlot.cxx
  src/MarkovChain.cxx
  src/MCMCCalculator.cxx
  src/MCMCInterval.cxx
  src/MCMCIntervalPlot.cxx
  src/MetropolisHastings.cxx
  src/NeymanConstruction.cxx
  src/NumberCountingPdfFactory.cxx
  src/NumberCountingUtils.cxx
  src/PdfProposal.cxx
  src/PointSetInterval.cxx
  src/ProfileInspector.cxx
  src/ProfileLikelihoodCalculator.cxx
  src/ProfileLikelihoodTestStat.cxx
  src/ProposalHelper.cxx
  src/RatioOfProfiledLikelihoodsTestStat.cxx
  src/RooStatsUtils.cxx
  src/SamplingDistPlot.cxx
  src/SamplingDistribution.cxx
  src/SequentialProposal.cxx
  src/SimpleInterval.cxx
  src/SimpleLikelihoodRatioTestStat.cxx
  src/SPlot.cxx
  src/ToyMCImportanceSampler.cxx
  src/ToyMCSampler.cxx
  src/UniformProposal.cxx
  src/UpperLimitMCSModule.cxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooStats/AsymptoticCalculator.h
    inc/RooStats/BayesianCalculator.h
    inc/RooStats/BernsteinCorrection.h
    inc/RooStats/CombinedCalculator.h
    inc/RooStats/ConfidenceBelt.h
    inc/RooStats/ConfInterval.h
    inc/RooStats/DetailedOutputAggregator.h
    inc/RooStats/FeldmanCousins.h
    inc/RooStats/FrequentistCalculator.h
    inc/RooStats/Heaviside.h
    inc/RooStats/HybridCalculator.h
    inc/RooStats/HybridResult.h
    inc/RooStats/HypoTestCalculatorGeneric.h
    inc/RooStats/HypoTestCalculator.h
    inc/RooStats/HypoTestInverter.h
    inc/RooStats/HypoTestInverterPlot.h
    inc/RooStats/HypoTestInverterResult.h
    inc/RooStats/HypoTestPlot.h
    inc/RooStats/HypoTestResult.h
    inc/RooStats/IntervalCalculator.h
    inc/RooStats/LikelihoodInterval.h
    inc/RooStats/LikelihoodIntervalPlot.h
    inc/RooStats/MarkovChain.h
    inc/RooStats/MaxLikelihoodEstimateTestStat.h
    inc/RooStats/MCMCCalculator.h
    inc/RooStats/MCMCInterval.h
    inc/RooStats/MCMCIntervalPlot.h
    inc/RooStats/MetropolisHastings.h
    inc/RooStats/ModelConfig.h
    inc/RooStats/NeymanConstruction.h
    inc/RooStats/NumberCountingPdfFactory.h
    inc/RooStats/NumberCountingUtils.h
    inc/RooStats/NumEventsTestStat.h
    inc/RooStats/PdfProposal.h
    inc/RooStats/PointSetInterval.h
    inc/RooStats/ProfileInspector.h
    inc/RooStats/ProfileLikelihoodCalculator.h
    inc/RooStats/ProfileLikelihoodTestStat.h
    inc/RooStats/ProposalFunction.h
    inc/RooStats/ProposalHelper.h
    inc/RooStats/RatioOfProfiledLikelihoodsTestStat.h
    inc/RooStats/RooStatsUtils.h
    inc/RooStats/SamplingDistPlot.h
    inc/RooStats/SamplingDistribution.h
    inc/RooStats/SequentialProposal.h
    inc/RooStats/SimpleInterval.h
    inc/RooStats/SimpleLikelihoodRatioTestStat.h
    inc/RooStats/SPlot.h
    inc/RooStats/TestStatistic.h
    inc/RooStats/TestStatSampler.h
    inc/RooStats/ToyMCImportanceSampler.h
    inc/RooStats/ToyMCSampler.h
    inc/RooStats/UniformProposal.h
    inc/RooStats/UpperLimitMCSModule.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)

target_link_libraries(RooStats
PUBLIC
Core
  RooFit
  RooFitCore
  Tree
  RIO
  Hist
  Matrix
  MathCore
  Graf
  Gpad
PRIVATE
  Minuit
  Foam
)


set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_GENERATE_TARGET_DICTIONARY(RooStats "-writeEmptyRootPCM" ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(RooStats)

ROOT_ADD_TEST_SUBDIRECTORY(test)
