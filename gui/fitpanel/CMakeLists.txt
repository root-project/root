############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanel package
############################################################################

if(root7)
    set(FitPanel_v7_src v7/src/)
    ROOT_GLOB_HEADERS(FitPanel_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
    set(FitPanel_v7_depend ROOTWebDisplay ROOTHistDraw)
endif()


ROOT_STANDARD_LIBRARY_PACKAGE(FitPanel 
                              HEADERS *.h ${FitPanel_v7_dict_headers}
                              SOURCES *.cxx ${FitPanel_v7_src}
                              DEPENDENCIES Gui Gpad Hist Graf Tree MathCore ${FitPanel_v7_depend} 
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM")
