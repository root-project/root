if(NOT pyroot OR NOT dataframe)
    return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTMLDataLoader
  HEADERS
    ROOT/ML/RDataLoaderEngine.hxx
    ROOT/ML/RFlat2DMatrix.hxx
  SOURCES
    src/RBatchLoader.cxx
    src/RBatchSink.cxx
    src/RClusterLoader.cxx
    src/RDataLoaderEngine.cxx
    src/RDatasetLoader.cxx
    src/RFlat2DMatrixOperators.cxx
    src/RSampler.cxx
  DEPENDENCIES
    ROOTDataFrame
)
