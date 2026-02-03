if(NOT pyroot OR NOT dataframe)
    return()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTMLDataLoader
  HEADERS
    ROOT/ML/RBatchGenerator.hxx
    ROOT/ML/RBatchLoader.hxx
    ROOT/ML/RChunkLoader.hxx
    ROOT/ML/RChunkConstructor.hxx
    ROOT/ML/RFlat2DMatrix.hxx
    ROOT/ML/RFlat2DMatrixOperators.hxx
    ROOT/ML/RDatasetLoader.hxx
    ROOT/ML/RSampler.hxx
  NO_SOURCES
  DEPENDENCIES
    ROOTDataFrame
)
