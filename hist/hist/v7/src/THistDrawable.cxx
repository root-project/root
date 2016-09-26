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

#include "ROOT/THistDrawable.hxx"

#include "ROOT/THistImpl.hxx"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TSystem.h" // TSystem::Load

#include <cassert>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::Internal;

void ROOT::Experimental::Internal::LoadHistPainterLibrary() {
  gSystem->Load("libHistPainter");
}


template <int DIMENSION>
THistPainterBase<DIMENSION>::~THistPainterBase() { fgPainter = nullptr; }

template <int DIMENSION>
THistPainterBase<DIMENSION>* THistPainterBase<DIMENSION>::fgPainter = nullptr;


Internal::THistDrawableBase::THistDrawableBase() = default;
Internal::THistDrawableBase::THistDrawableBase(THistDrawableBase&&) = default;
Internal::THistDrawableBase::~THistDrawableBase() = default;

Internal::THistDrawableBase&
Internal::THistDrawableBase::operator=(THistDrawableBase&&) = default;


template <int DIMENSIONS>
bool
THistDrawable<DIMENSIONS>::UpdateOldHist() {
  auto implBase = fHistImpl.Get();
  if (!implBase) {
    fOldHist.reset();
    return false;
  }

  std::array<TAxisView, DIMENSIONS> axes;
  for (int i = 0; i < DIMENSIONS; ++i)
    axes[i] = implBase->GetAxis(i);

  TH1 *old = nullptr;

  // Create a unique name, for what it's worth.
  std::string histName;
  {
    std::stringstream strm;
    strm << "drawAdaptor" << this;
  }

  // Create old histogram; set nbins because TH1::fNcells is not accessible.
  switch (DIMENSIONS) {
    case 1:
      old = new TH1D(histName.c_str(),
                     implBase->GetTitle().c_str(),
                     axes[0].GetNBins() - 2, 0., 1.);
      break;
    case 2:
      old = new TH2D(histName.c_str(),
                     implBase->GetTitle().c_str(),
                     axes[0].GetNBins() - 2, 0., 1.,
                     axes[1].GetNBins() - 2, 0., 1.);
      break;
    case 3:
      old = new TH3D(histName.c_str(),
                     implBase->GetTitle().c_str(),
                     axes[0].GetNBins() - 2, 0., 1.,
                     axes[1].GetNBins() - 2, 0., 1.,
                     axes[2].GetNBins() - 2, 0., 1.);
      break;
    default:
      // And anyway, this should really give a missing symbol due to the export
      // template.
      R__ERROR_HERE("Hist") << "Drawing of " << DIMENSIONS
                            << " dimensional histograms not supported.";
      return false;
  }

  old->SetDirectory(nullptr);

  // See TH1::SetBins().
  std::array<TAxis*, 3> oldAxes{{old->GetXaxis(), old->GetYaxis(), old->GetZaxis()}};
  for (int i = 0; i < DIMENSIONS; ++i) {
    oldAxes[i]->SetRange(0, 0);
    oldAxes[i]->SetTitle(axes[i].GetTitle().c_str());
    if (axes[i].GetAsEquidistant()) {
      oldAxes[i]->Set(axes[i].GetNBins() - 2, axes[i].GetFrom(), axes[i].GetTo());
    } else if (const TAxisIrregular* irr = axes[i].GetAsIrregular()) {
      oldAxes[i]->Set(axes[i].GetNBins() - 2, &irr->GetBinBorders()[0]);
    } else {
      assert(0 && "Logic error; the axis is neither equidistant nor irregular.");
    }
  }

  int nBins = implBase->GetNBins();
  old->SetBinsLength(nBins);
  if (implBase->HasBinUncertainty())
    old->Sumw2();

  // Set the bin content + uncertainty.
  for (int binidx = 0; binidx < nBins; ++binidx) {
    old->SetBinContent(binidx, implBase->GetBinContentAsDouble(binidx));
    old->SetBinError(binidx, implBase->GetBinUncertainty(binidx));
  }
  fOldHist.reset(old);
  return true;
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
