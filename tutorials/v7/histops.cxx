/// \file histops.cxx
/// \ingroup Tutorials
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice, it might do evil. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/THist.h"
#include <iostream>

void histops() {

  // Create a 2D histogram with an X axis with equidistant bins, and a y axis
  // with irregular binning.
  ROOT::TH2D hist1({100, 0., 1.}, {{0., 1., 2., 3.,10.}});

  // Fill weight 1. at the coordinate 0.01, 1.02.
  hist1.Fill({0.01, 1.02});


  ROOT::TH2D hist2({{ {10, 0., 1.}, {{0., 1., 2., 3.,10.}} }});
  // Fill weight 1. at the coordinate 0.01, 1.02.
  hist2.Fill({0.01, 1.02});

  ROOT::Add(hist1, hist2);

  int binidx = hist1.GetImpl()->GetBinIndex({0.01, 1.02});
  std::cout << hist1.GetImpl()->GetBinContent(binidx) << std::endl;
}
