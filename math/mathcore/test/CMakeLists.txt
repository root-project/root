project(mathcore-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread)

set(TestSource
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
    fit/testFit.cxx
    fit/testGraphFit.cxx
    fit/SparseDataComparer.cxx
    fit/SparseFit4.cxx
    fit/SparseFit3.cxx )


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
  ROOT_ADD_TEST(mathcore-${testname} COMMAND ${testname})
endforeach()

