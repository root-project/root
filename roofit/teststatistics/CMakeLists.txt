############################################################################
# CMakeLists.txt file for building ROOT RooFitTestStatistics package
# @author Patrick Bos, Netherlands eScience Center
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitTestStatistics
    HEADERS
        RooFit/TestStatistics/RooAbsL.h
        RooFit/TestStatistics/RooUnbinnedL.h
        RooFit/TestStatistics/RooBinnedL.h
        RooFit/TestStatistics/RooMultiL.h
        RooFit/TestStatistics/RooConstraintL.h
        RooFit/TestStatistics/RooRealL.h
        RooFit/TestStatistics/LikelihoodWrapper.h
        RooFit/TestStatistics/LikelihoodJob.h
        RooFit/TestStatistics/LikelihoodGradientWrapper.h
        RooFit/TestStatistics/LikelihoodGradientJob.h
    SOURCES
        src/RooAbsL.cxx
        src/RooUnbinnedL.cxx
        src/RooBinnedL.cxx
        src/RooMultiL.cxx
        src/RooConstraintL.cxx
        src/RooRealL.cxx
        src/LikelihoodWrapper.cxx
        src/LikelihoodJob.cxx
        src/LikelihoodGradientWrapper.cxx
        src/LikelihoodGradientJob.cxx
    DICTIONARY_OPTIONS
        "-writeEmptyRootPCM"
    DEPENDENCIES
        RooFitMultiProcess
        RooFitCore
)

if(testing)
    ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
