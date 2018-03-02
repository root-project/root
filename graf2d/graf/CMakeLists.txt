############################################################################
# CMakeLists.txt file for building ROOT graf2d/graf package
# @author Pere Mato, CERN
############################################################################

include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(Graf
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${FREETYPE_LIBRARIES} ${ZLIB_LIBRARIES} mathtext
                              DEPENDENCIES Hist Matrix MathCore RIO BUILTINS FREETYPE)

