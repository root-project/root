include(headers.cmake)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTHist
  HEADERS
    ${histv7_headers}
  NO_SOURCES
  DEPENDENCIES
    Core
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
