# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Regression test for https://github.com/root-project/root/issues/23491
ROOT_ADD_GTEST(testTGraphErrorsBand test_TGraphErrors_band.cxx LIBRARIES HistPainter Hist Gpad)
