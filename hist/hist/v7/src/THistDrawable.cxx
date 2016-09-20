/// \file THistDrawable.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/THistDrawable.h"

#include "ROOT/THistImpl.h"

#include "TH2.h"
#include "TSystem.h" // TSystem::Load

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

void ROOT::Experimental::Internal::LoadHistPainterLibrary() {
  gSystem->Load("libHistPainter");
}


template <int DIMENSION>
THistPainterBase<DIMENSION>::~THistPainterBase() { fgPainter = nullptr; }

template <int DIMENSION>
THistPainterBase<DIMENSION>* THistPainterBase<DIMENSION>::fgPainter = nullptr;

template <int DIMENSIONS>
bool
THistDrawable<DIMENSIONS>::UpdateOldHist() {
  if (auto implBase = fHistImpl.lock()) {
    TH2D *old = new TH2D("drawAdaptor", "tmp",
                         implBase->GetAxis(0).GetNBins() - 2,
                         implBase->GetAxis(0).GetFrom(),
                         implBase->GetAxis(0).GetTo(),
                         implBase->GetAxis(1).GetNBins() - 2,
                         implBase->GetAxis(1).GetFrom(),
                         implBase->GetAxis(1).GetTo());
    old->SetDirectory(nullptr);
    int nBins = implBase->GetNBins();
    for (int binidx = 0; binidx < nBins; ++binidx) {
      old->SetBinContent(binidx, implBase->GetBinContentAsDouble(binidx));
      old->SetBinError(binidx, implBase->GetBinUncertainty(binidx));
    }
    fOldHist.reset(old);
    return true;
  }

  fOldHist.reset();
  return false;
}


namespace ROOT {
namespace Experimental {
namespace Internal {
template class THistPainterBase<1>;
template class THistPainterBase<2>;
template class THistPainterBase<3>;

template class THistDrawable<1>;
template class THistDrawable<2>;
template class THistDrawable<3>;
}
}
}
