############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SpectrumPainter
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Graf Hist)
