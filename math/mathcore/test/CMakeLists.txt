project(mathcore-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread)

set(TestSource
    testMathRandom.cxx
    testTMath.cxx
    testBinarySearch.cxx
    testSortOrder.cxx
    stressTMath.cxx
    stressTF1.cxx
    testIntegration.cxx
    testRootFinder.cxx
    testSampleQuantiles.cxx
    kDTreeTest.cxx
    testkdTreeBinning.cxx
    newKDTreeTest.cxx
    binarySearchTime.cxx
    stdsort.cxx
    testSpecFuncErf.cxx
    testSpecFuncGamma.cxx
    testSpecFuncBeta.cxx
    testSpecFuncBetaI.cxx
    testSpecFuncSiCi.cxx
    testIntegrationMultiDim.cxx
    testAnalyticalIntegrals.cxx
    testTStatistic.cxx
    testKahan.cxx
    fit/testFit.cxx
    fit/testGraphFit.cxx
    fit/SparseDataComparer.cxx
    fit/SparseFit4.cxx
    fit/SparseFit3.cxx
    fit/testBinnedFitExecPolicy.cxx
    fit/testLogLExecPolicy.cxx )

set(testMathRandom_LABELS longtest)
set(testFitPerf_LABELS longtest)

if(ROOT_roofit_FOUND)
  list(APPEND TestSource fit/testRooFit.cxx)
  list(APPEND Libraries RooFit)
  if(ROOT_mathmore_FOUND)
    list(APPEND TestSource fit/testFitPerf.cxx)
  endif()
endif()

if(ROOT_mathmore_FOUND)
  list(APPEND TestSource stressGoFTest.cxx)
  list(APPEND Libraries MathMore)
endif()

if(ROOT_minuit2_FOUND)
  list(APPEND TestSource fit/testMinim.cxx)
  list(APPEND Libraries Minuit2)
endif()


#---Build and add all the defined test in the list---------------
foreach(file ${TestSource})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES ${Libraries})
  ROOT_ADD_TEST(mathcore-${testname} COMMAND ${testname} LABELS ${${testname}_LABELS})
endforeach()


ROOT_GENERATE_DICTIONARY(G__TrackMathCoreUnitDict ${CMAKE_CURRENT_SOURCE_DIR}/stress/TrackMathCore.h MODULE TrackMathCoreUnitDict LINKDEF stress/TrackMathCoreLinkDef.h DEPENDENCIES Core MathCore RIO GenVector)
ROOT_LINKER_LIBRARY(TrackMathCoreUnitDict G__TrackMathCoreUnitDict.cxx LIBRARIES Core MathCore RIO GenVector)

ROOT_ADD_GTEST(stressMathCoreUnit stress/testSMatrix.cxx stress/testGenVector.cxx
        stress/testStat.cxx stress/testSVector.cxx stress/testVector.cxx stress/testVector34.cxx
        stress/TestHelper.cxx
        LIBRARIES Core MathCore Hist RIO Tree GenVector)
