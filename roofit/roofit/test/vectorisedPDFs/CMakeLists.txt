if(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(DEFINED ENV{VTUNE_AMPLIFIER_2019_DIR})
    set(VTUNE_DIR ENV{VTUNE_AMPLIFIER_2019_DIR})
  elseif(DEFINED ENV{VTUNE_PROFILER_2020_DIR})
    set(VTUNE_DIR ENV{VTUNE_PROFILER_2020_DIR})
  endif()
  # To be able to start/stop the vtune profiler, ittnotify must be available.
  target_include_directories(VectorisedPDFTests PUBLIC "${VTUNE_DIR}/include/")
  target_link_libraries(VectorisedPDFTests INTERFACE "${VTUNE_DIR}/lib64/libittnotify.a")
endif()

add_library(VectorisedPDFTests STATIC VectorisedPDFTests.cxx)
target_link_libraries(VectorisedPDFTests PUBLIC ROOT::Gpad ROOT::RooFitCore ROOT::RooFit GTest::gtest)

if(mathmore)
  set(extra_sources testLegendre.cxx)
  set(extra_libraries ROOT::RooFitMore)
endif()

ROOT_ADD_GTEST(testVectorizedPDFs
    testAddPdf.cxx
    testArgusBG.cxx
    testBernstein.cxx
    testBifurGauss.cxx
    testBreitWigner.cxx
    testBukin.cxx
    testCBShape.cxx
    testChebychev.cxx
    testChiSquarePdf.cxx
    testCompatMode.cxx
    testDstD0BG.cxx
    testExponential.cxx
    testGamma.cxx
    testGauss.cxx
    testJohnson.cxx
    testLandau.cxx
    testLognormal.cxx
    testNestedPDFs.cxx
    testNovosibirsk.cxx
    testPoisson.cxx
    testPolynomial.cxx
    testProductPdf.cxx
    testVoigtian.cxx
    ${extra_sources}
  LIBRARIES
    VectorisedPDFTests
    ${extra_libraries}
)

ROOT_ADD_GTEST(testGaussBinned testGaussBinned.cxx
              LIBRARIES ROOT::Gpad ROOT::RooFitCore ROOT::RooFit)

