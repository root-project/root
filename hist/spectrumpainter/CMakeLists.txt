############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SpectrumPainter
  HEADERS
    TSpectrum2Painter.h
  SOURCES
    src/TSpectrum2Painter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Graf
    Hist
)
