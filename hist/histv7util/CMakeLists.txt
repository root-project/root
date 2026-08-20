ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHistUtil
  HEADERS
    ROOT/Hist/ConversionUtils.hxx
    ROOT/Hist/ConvertToTH1.hxx
    ROOT/Hist/ConvertToTH2.hxx
    ROOT/Hist/ConvertToTH3.hxx
  SOURCES
    ConversionUtils.cxx
    ConvertToTH1.cxx
    ConvertToTH2.cxx
    ConvertToTH3.cxx
  DEPENDENCIES
    Core
    Hist
    ROOTHist
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
