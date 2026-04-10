if(NOT pyroot OR NOT dataframe)
    return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTMLDataLoader
  HEADERS
    ROOT/ML/RDataLoaderEngine.hxx
    ROOT/ML/RBatchLoader.hxx
    ROOT/ML/RClusterLoader.hxx
    ROOT/ML/RFlat2DMatrix.hxx
    ROOT/ML/RFlat2DMatrixOperators.hxx
    ROOT/ML/RDatasetLoader.hxx
    ROOT/ML/RSampler.hxx
  SOURCES
    src/RBatchLoader.cxx
    src/RFlat2DMatrixOperators.cxx
    src/RSampler.cxx
  DEPENDENCIES
    ROOTDataFrame
)
