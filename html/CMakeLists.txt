############################################################################
# CMakeLists.txt file for building ROOT html package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Html
  HEADERS
    TClassDocOutput.h
    TDocDirective.h
    TDocInfo.h
    TDocOutput.h
    TDocParser.h
    THtml.h
  SOURCES
    src/TClassDocOutput.cxx
    src/TDocDirective.cxx
    src/TDocInfo.cxx
    src/TDocOutput.cxx
    src/TDocParser.cxx
    src/THtml.cxx
  DEPENDENCIES
    Graf
    Thread
)
