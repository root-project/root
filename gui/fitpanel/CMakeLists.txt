############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanel package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FitPanel 
                              HEADERS *.h 
                              SOURCES *.cxx 
                              DEPENDENCIES Gui Gpad Hist Graf Tree MathCore 
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM")
