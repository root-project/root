/// \file draw.cxx
/// \ingroup Tutorials
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice, it might do evil. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Rtypes.h"

R__LOAD_LIBRARY(libGpad)

#include "ROOT/THist.h"
#include "ROOT/TCanvas.h"
#include "ROOT/TDirectory.h"
#include <iostream>

void example() {
  using namespace ROOT;

  auto pHist = MakeCoop<THist<2, double>>(TAxisConfig{100, 0., 1.},
                                          TAxisConfig{{0., 1., 2., 3.,10.}});

  pHist->Fill({0.01, 1.02});
  ROOT::TDirectory::Heap().Add("hist", pHist);

  auto canvas = ROOT::TCanvas::Create("MyCanvas");
  canvas->Draw(pHist);
}

void draw() {
  example();

  // And the event loop (?) will call
  for (auto&& canv: ROOT::TCanvas::GetCanvases())
    canv->Paint();
}
