############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Hbook
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES minicern
                              DEPENDENCIES Hist Matrix Tree Graf TreePlayer RIO)
