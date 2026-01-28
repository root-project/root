ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistUtil
  HEADERS
    ROOT/Hist/ConversionUtils.hxx
    ROOT/Hist/ConvertToTH1.hxx
  SOURCES
    ConversionUtils.cxx
    ConvertToTH1.cxx
  DEPENDENCIES
    Core
    Hist
    ROOTHist
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
