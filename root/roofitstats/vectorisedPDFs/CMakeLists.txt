if(CMAKE_CXX_COMPILER_ID STREQUAL Intel AND DEFINED ENV{VTUNE_AMPLIFIER_2019_DIR})
  # To be able to start/stop the vtune profiler, ittnotify must be available.
  include_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/include/")
  link_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/lib64/")
  link_libraries(-littnotify)
endif()

add_library(VectorisedPDFTests STATIC VectorisedPDFTests.cxx)
link_libraries(VectorisedPDFTests Gpad RooFitCore RooFit)
add_dependencies(VectorisedPDFTests RooFitCore)

ROOT_ADD_GTEST(testCompatMode testCompatMode.cxx)
ROOT_ADD_GTEST(testGauss testGauss.cxx)
ROOT_ADD_GTEST(testPoisson testPoisson.cxx)
ROOT_ADD_GTEST(testAddPdf testAddPdf.cxx)
ROOT_ADD_GTEST(testExponential testExponential.cxx)
ROOT_ADD_GTEST(testNestedPDFs testNestedPDFs.cxx
              WILLFAIL)
ROOT_ADD_GTEST(testProductPdf testProductPdf.cxx)
ROOT_ADD_GTEST(testJohnson testJohnson.cxx)
ROOT_ADD_GTEST(testLandau testLandau.cxx)
ROOT_ADD_GTEST(testBukin testBukin.cxx)    
ROOT_ADD_GTEST(testChebychev testChebychev.cxx)    
ROOT_ADD_GTEST(testPolynomial testPolynomial.cxx
              WILLFAIL)    
ROOT_ADD_GTEST(testBernstein testBernstein.cxx)    
ROOT_ADD_GTEST(testArgusBG testArgusBG.cxx
              WILLFAIL)    
ROOT_ADD_GTEST(testBifurGauss testBifurGauss.cxx
              WILLFAIL)    
ROOT_ADD_GTEST(testBreitWigner testBreitWigner.cxx)    
ROOT_ADD_GTEST(testCBShape testCBShape.cxx)    
ROOT_ADD_GTEST(testGamma testGamma.cxx
              WILLFAIL)    
ROOT_ADD_GTEST(testLegendre testLegendre.cxx)    
ROOT_ADD_GTEST(testChiSquarePdf testChiSquarePdf.cxx
              WILLFAIL)    
