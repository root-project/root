############################################################################
# CMakeLists.txt file for building ROOT gui/recorder package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Recorder
  HEADERS
    TRecorder.h
  SOURCES
    src/TRecorder.cxx
  DEPENDENCIES
    RIO
    Tree
    Gui
    Thread
)
