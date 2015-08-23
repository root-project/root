############################################################################
# CMakeLists.txt file for building ROOT hist/spectrum package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Spectrum DEPENDENCIES Hist Matrix DICTIONARY_OPTIONS "-writeEmptyRootPCM")
