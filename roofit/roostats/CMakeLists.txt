############################################################################
# CMakeLists.txt file for building ROOT roostats package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooStats
                              HEADERS RooStats/*.h
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core RooFit RooFitCore Tree RIO Hist Matrix
                                           MathCore Minuit Foam Graf Gpad)
