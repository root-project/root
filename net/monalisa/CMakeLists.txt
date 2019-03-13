############################################################################
# CMakeLists.txt file for building ROOT net/monalisa package
# @author Pere Mato, CERN
############################################################################

include_directories(${MONALISA_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(Monalisa
  HEADERS
    TMonaLisaWriter.h
  SOURCES
    src/TMonaLisaWriter.cxx
  LIBRARIES
    ${MONALISA_LIBRARIES}
  DEPENDENCIES
    Net
    RIO
)
