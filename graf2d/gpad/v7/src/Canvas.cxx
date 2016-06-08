/// \file Canvas.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-10
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/Canvas.h"

#include "ROOT/TDrawable.h"

#include <memory>

void ROOT::Experimental::TCanvas::Paint() {
  for (auto&& drw: fPrimitives) {
    drw->Paint();
  }
}

namespace ROOT {
namespace Experimental {
namespace Internal {
class TCanvasSharedPtrMaker: public ROOT::Experimental::TCanvas {
public:
  TCanvasSharedPtrMaker() {}
};
}
}
}

namespace {
static
std::vector<std::weak_ptr<ROOT::Experimental::TCanvas>>& GetHeldCanvases() {
  static std::vector<std::weak_ptr<ROOT::Experimental::TCanvas>> sCanvases;
  return sCanvases;
}
};

const std::vector<std::weak_ptr<ROOT::Experimental::TCanvas>> &
ROOT::Experimental::TCanvas::GetCanvases() {
  return GetHeldCanvases();
}

ROOT::Experimental::TCanvasPtr ROOT::Experimental::TCanvas::Create(
   std::experimental::string_view /*name*/) {
  // TODO: name registration (TDirectory?)
  auto pCanvas = std::make_shared<Internal::TCanvasSharedPtrMaker>();
  GetHeldCanvases().emplace_back(pCanvas);
  return ROOT::Experimental::TCanvasPtr(pCanvas);
}
