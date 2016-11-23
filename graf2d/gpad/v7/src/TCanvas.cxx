/// \file TCanvas.cxx
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

#include "ROOT/TCanvas.hxx"

#include "ROOT/TDrawable.hxx"
#include "TCanvas.h"
#include "TROOT.h"

#include "ROOT/TLogger.hxx"

#include <memory>

namespace {
static
std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>>& GetHeldCanvases() {
  static std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> sCanvases;
  return sCanvases;
}

}

namespace ROOT {
namespace Experimental {
namespace Internal {

class TV5CanvasAdaptor: public TObject {
  ROOT::Experimental::TCanvas& fNewCanv;
  ::TCanvas* fOldCanv; // ROOT owns them.

public:
  /// Construct an old TCanvas, append TV5CanvasAdaptor to its primitives.
  /// That way, TV5CanvasAdaptor::Paint() is called when the TCanvas paints its
  /// primitives, and TV5CanvasAdaptor::Paint() can forward to
  /// Experimental::TCanvas::Paint().
  TV5CanvasAdaptor(ROOT::Experimental::TCanvas& canv):
    fNewCanv(canv),
    fOldCanv(new ::TCanvas())
  {
    fOldCanv->SetTitle(canv.GetTitle().c_str());
    AppendPad();
  }

  ~TV5CanvasAdaptor() {
    // Make sure static destruction hasn't already destroyed the old TCanvases.
    if (gROOT && gROOT->GetListOfCanvases() && !gROOT->GetListOfCanvases()->IsEmpty())
      fOldCanv->RecursiveRemove(this);
  }

  void Paint(Option_t */*option*/="") override {
    fNewCanv.Paint();
  }
};
}
}
}

const std::vector<std::shared_ptr<ROOT::Experimental::TCanvas>> &
ROOT::Experimental::TCanvas::GetCanvases() {
  return GetHeldCanvases();
}


ROOT::Experimental::TCanvas::TCanvas() {
  fAdaptor = std::make_unique<Internal::TV5CanvasAdaptor>(*this);
}

ROOT::Experimental::TCanvas::~TCanvas() = default;

void ROOT::Experimental::TCanvas::Paint() {
  for (auto&& drw: fPrimitives) {
    drw->Paint(*this);
  }
}

std::shared_ptr<ROOT::Experimental::TCanvas>
ROOT::Experimental::TCanvas::Create(const std::string& title) {
  auto pCanvas = std::make_shared<TCanvas>();
  pCanvas->SetTitle(title);
  GetHeldCanvases().emplace_back(pCanvas);
  return pCanvas;
}

// TODO: removal from GetHeldCanvases().
