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
if(MSVC)
  target_link_libraries(VectorisedPDFTests PUBLIC gtest Gpad RooFitCore RooFit)
else()
  target_link_libraries(VectorisedPDFTests PUBLIC gtest ROOT::Gpad ROOT::RooFitCore ROOT::RooFit)
endif()

ROOT_ADD_GTEST(testCompatMode testCompatMode.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testGauss testGauss.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testPoisson testPoisson.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testAddPdf testAddPdf.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testExponential testExponential.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testNestedPDFs testNestedPDFs.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testProductPdf testProductPdf.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testJohnson testJohnson.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testLandau testLandau.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testBukin testBukin.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testChebychev testChebychev.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testPolynomial testPolynomial.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testBernstein testBernstein.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testArgusBG testArgusBG.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testBifurGauss testBifurGauss.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testBreitWigner testBreitWigner.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testCBShape testCBShape.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testGamma testGamma.cxx
              LIBRARIES VectorisedPDFTests)
if(ROOT_mathmore_FOUND)
  if(MSVC)
    ROOT_ADD_GTEST(testLegendre testLegendre.cxx
                  LIBRARIES VectorisedPDFTests RooFitMore)
  else()
    ROOT_ADD_GTEST(testLegendre testLegendre.cxx
                  LIBRARIES VectorisedPDFTests ROOT::RooFitMore)
  endif()
endif()
ROOT_ADD_GTEST(testChiSquarePdf testChiSquarePdf.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testDstD0BG testDstD0BG.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testLognormal testLognormal.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testNovosibirsk testNovosibirsk.cxx
              LIBRARIES VectorisedPDFTests)
ROOT_ADD_GTEST(testVoigtian testVoigtian.cxx
              LIBRARIES VectorisedPDFTests)
if(MSVC)
  ROOT_ADD_GTEST(testGaussBinned testGaussBinned.cxx
                LIBRARIES Gpad RooFitCore RooFit)
else()
  ROOT_ADD_GTEST(testGaussBinned testGaussBinned.cxx
                LIBRARIES ROOT::Gpad ROOT::RooFitCore ROOT::RooFit)
endif()

