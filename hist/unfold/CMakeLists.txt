############################################################################
# CMakeLists.txt file for building ROOT hist/unfold package
############################################################################

set(libname Unfold)

ROOT_STANDARD_LIBRARY_PACKAGE(Unfold DEPENDENCIES Hist XMLParser Matrix DICTIONARY_OPTIONS "-writeEmptyRootPCM")