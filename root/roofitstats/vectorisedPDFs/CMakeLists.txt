if(CMAKE_CXX_COMPILER_ID STREQUAL Intel AND DEFINED ENV{VTUNE_AMPLIFIER_2019_DIR})
  # To be able to start/stop the vtune profiler, ittnotify must be available.
  include_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/include/")
  link_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/lib64/")
  link_libraries(-littnotify)
endif()

add_library(VectorisedPDFTests STATIC VectorisedPDFTests.cxx)
target_link_libraries(VectorisedPDFTests PUBLIC gtest ROOT::Gpad ROOT::RooFitCore ROOT::RooFit)

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
              LIBRARIES VectorisedPDFTests
              WILLFAIL)
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
   ROOT_ADD_GTEST(testLegendre testLegendre.cxx
                 LIBRARIES VectorisedPDFTests ROOT::RooFitMore)
endif()
ROOT_ADD_GTEST(testChiSquarePdf testChiSquarePdf.cxx
              LIBRARIES VectorisedPDFTests)
