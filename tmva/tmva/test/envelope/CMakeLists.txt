############################################################################
# CMakeLists.txt file for building Envelope tests.
# @author  Omar Zapata
############################################################################
ROOT_ADD_GTEST(TMVA-Envelope testEnvelope.cxx LIBRARIES Core TMVA RIO)
ROOT_ADD_GTEST(TMVA-Classification testClassification.cxx LIBRARIES Core TMVA RIO)
