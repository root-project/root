############################################################################
# CMakeLists.txt file for building ROOT hist/hist package
############################################################################

set(libname Hist)

if(root7)
    set(root7src v7/src/RAxis.cxx)
    set(Hist_v7_dict_headers
            v7/inc/ROOT/RAxis.hxx
            v7/inc/ROOT/RHist.hxx
            v7/inc/ROOT/RHistBinIter.hxx
            v7/inc/ROOT/RHistBufferedFill.hxx
            v7/inc/ROOT/RHistConcurrentFill.hxx
            v7/inc/ROOT/RHistData.hxx
            v7/inc/ROOT/RHistImpl.hxx
            v7/inc/ROOT/RHistUtils.hxx
            v7/inc/ROOT/RHistView.hxx
       )
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Hist
                              HEADERS *.h Math/*.h v5/*.h ${Hist_v7_dict_headers}
                              SOURCES *.cxx ${root7src}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Matrix MathCore RIO)

ROOT_ADD_TEST_SUBDIRECTORY(test)

if(root7)
    ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
endif()
