if(CMAKE_CXX_COMPILER_ID STREQUAL Intel AND DEFINED ENV{VTUNE_AMPLIFIER_2019_DIR})
  # To be able to start/stop the vtune profiler, ittnotify must be available.
  include_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/include/")
  link_directories("$ENV{VTUNE_AMPLIFIER_2019_DIR}/lib64/")
  link_libraries(-littnotify)
endif()

add_library(VectorisedPDFTests STATIC VectorisedPDFTests.cxx)
link_libraries(VectorisedPDFTests Gpad RooFitCore RooFit)

ROOT_ADD_GTEST(testCompatMode testCompatMode.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testGauss testGauss.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testPoisson testPoisson.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testAddPdf testAddPdf.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testExponential testExponential.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testNestedPDFs testNestedPDFs.cxx
              LIBRARIES Gpad RooFitCore RooFit
              WILLFAIL)
ROOT_ADD_GTEST(testProductPdf testProductPdf.cxx
              LIBRARIES Gpad RooFitCore RooFit)
ROOT_ADD_GTEST(testJohnson testJohnson.cxx
              LIBRARIES Gpad RooFitCore RooFit)

