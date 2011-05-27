############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/graf)

ROOT_STANDARD_LIBRARY_PACKAGE(Postscript DEPENDENCIES Graf)

