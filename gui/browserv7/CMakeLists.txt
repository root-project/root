############################################################################
# CMakeLists.txt file for building ROOT gui/browserv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowserv7
  HEADERS
    ROOT/RBrowserItem.hxx
    ROOT/RBrowser.hxx
  SOURCES
    src/RBrowser.cxx
  DEPENDENCIES
    ROOTWebDisplay
)
