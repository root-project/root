############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

if(root7)
    set(root7src v7/src/RHistPainter.cxx)
    set(HISTDRAWLIB ROOTHistDraw)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(HistPainter
                              SOURCES *.cxx ${root7src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES  Graf Hist Matrix MathCore ${HISTDRAWLIB} Gpad)

