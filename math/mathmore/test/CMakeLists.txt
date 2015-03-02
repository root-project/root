project(mathmore-tests)
find_package(ROOT REQUIRED)

include_directories(${ROOT_INCLUDE_DIRS})

set(Libraries Core RIO Net Hist Graf Graf3d Gpad Tree
              Rint Postscript Matrix Physics MathCore Thread MathMore)

set(TestMathMoreSource
      testChebyshev.cxx
     testPermute.cxx
     testDerivation.cxx
     testGSLIntegration.cxx
     testMCIntegration.cxx
     testGSLRootFinder.cxx
     testMinimization1D.cxx
     testInterpolation.cxx
     testRandom.cxx
     testRandomDist.cxx
     testQuasiRandom.cxx
     testSpecFunc.cxx
     testStatFunc.cxx
     testFunctor.cxx
     testVavilov.cxx
     simanTSP.cxx  )


if(ROOT_unuran_FOUND)
  list(APPEND Libraries Unuran)
  add_definitions(-DHAVE_UNURAN)
endif()


if(ROOT_minuit2_FOUND)
  list(APPEND TestSource fit/testMinim.cxx)
  list(APPEND Libraries Minuit2)
endif()


#some tests requires directly gsl
include_directories(${GSL_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
add_definitions(-DUSE_ROOT_ERROR -DHAVE_ROOTLIBS)

#---Build and add all the defined test in the list---------------
foreach(file ${TestMathMoreSource})
  get_filename_component(testname ${file} NAME_WE)
  ROOT_EXECUTABLE(${testname} ${file} LIBRARIES ${GSL_LIBRARIES} ${Libraries})
  ROOT_ADD_TEST(mathmore-${testname} COMMAND ${testname}) 
endforeach()

