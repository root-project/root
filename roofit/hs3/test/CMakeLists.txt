ROOT_ADD_PYUNITTEST(roofit-hs3-histfactory-json test_hs3_histfactory_json.py
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/test_hs3_histfactory_json_input.root
  )

ROOT_ADD_GTEST(testRooFitHS3 testRooFitHS3.cxx LIBRARIES RooFitCore RooFit RooFitHS3)
