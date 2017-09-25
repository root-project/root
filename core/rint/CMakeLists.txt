############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Rint
                              STAGE1
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core)
