/// \file TCanvas.cxx
/// \ingroup Gpad
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice, it might do evil. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TCanvas.h"

#include "ROOT/TDrawable.h"

void ROOT::TCanvas::Paint() {
  for (auto&& drw: fPrimitives) {
    drw->Paint();
  }
}

namespace {
static
std::vector<ROOT::TCoopPtr<ROOT::TCanvas>>& GetHeldCanvases() {
  static std::vector<ROOT::TCoopPtr<ROOT::TCanvas>> sCanvases;
  return sCanvases;
}
};

const std::vector<ROOT::TCoopPtr<ROOT::TCanvas>> &
ROOT::TCanvas::GetCanvases() {
  return GetHeldCanvases();
}

ROOT::TCoopPtr<ROOT::TCanvas> ROOT::TCanvas::Create(
   std::experimental::string_view name) {
  // TODO: name registration (TDirectory?)
  auto pCanvas = TCoopPtr<TCanvas>(new TCanvas());
  GetHeldCanvases().emplace_back(pCanvas);
  return pCanvas;
}