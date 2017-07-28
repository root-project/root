############################################################################
# CMakeLists.txt file for building ROOT montecarlo/eg package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(EG
                              HEADERS T*.h
                              DEPENDENCIES Graf3d Graf Gpad MathCore )
