############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Postscript
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES  ${ZLIB_LIBRARIES} mathtext
                              DEPENDENCIES Graf)
