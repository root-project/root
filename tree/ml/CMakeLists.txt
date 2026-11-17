if(NOT pyroot OR NOT dataframe)
    return()
endif()

set(MLDATALOADER_HEADERS
  ROOT/ML/RDataLoaderEngine.hxx
  ROOT/ML/RFlat2DMatrix.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTMLDataLoader
  HEADERS
    ${MLDATALOADER_HEADERS}
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

set(RELATIVE_INC_HEADERS ${MLDATALOADER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTMLDataLoader
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/RBatchLoader.hxx
    src/RBatchSink.hxx
    src/RClusterLoader.hxx
    src/RDatasetLoader.hxx
    src/RFlat2DMatrixOperators.hxx
    src/RSampler.hxx
)

