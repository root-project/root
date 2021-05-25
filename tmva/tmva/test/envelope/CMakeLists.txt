# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building Envelope tests.
# @author  Omar Zapata
############################################################################
ROOT_ADD_GTEST(TMVA-Envelope testEnvelope.cxx LIBRARIES Core TMVA RIO)
ROOT_ADD_GTEST(TMVA-Classification testClassification.cxx LIBRARIES Core TMVA RIO)
