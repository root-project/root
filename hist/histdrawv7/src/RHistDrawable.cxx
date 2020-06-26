/// \file RHistDrawable.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"

#include "ROOT/RHistDisplayItem.hxx"
#include "ROOT/RAxis.hxx"

using namespace ROOT::Experimental;

std::unique_ptr<RDisplayItem> RHist2Drawable::Display(const RDisplayContext &ctxt)
{
   if (!fOptimize)
      return RHistDrawable<2>::Display(ctxt);

   auto item = std::make_unique<RHistDisplayItem>(*this);

   auto himpl = fHistImpl.get();

   if (himpl) {

      int nbinsx = himpl->GetAxis(0).GetNBinsNoOver();
      int nbinsy = himpl->GetAxis(1).GetNBinsNoOver();

      auto &bins = item->GetBinContent();
      bins.resize(nbinsy*nbinsx);
      bins[0] = 0; // bin[0] does not used in RHist, Why?

      for (int ny = 0; ny < nbinsy; ++ny)
         for (int nx = 0; nx < nbinsx; ++nx) {
            bins[ny*nbinsx + nx] = himpl->GetBinContentAsDouble(ny*nbinsx + nx + 1);
         }

      item->AddAxis(&himpl->GetAxis(0), 0, nbinsx);
      item->AddAxis(&himpl->GetAxis(1), 0, nbinsy);
   }

   return item;
}
