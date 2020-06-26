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
#include "ROOT/RFrame.hxx"
#include "ROOT/RPadBase.hxx"
#include "ROOT/RAxis.hxx"

using namespace ROOT::Experimental;

std::unique_ptr<RDisplayItem> RHist2Drawable::CreateHistDisplay(const RDisplayContext &ctxt)
{
   auto item = std::make_unique<RHistDisplayItem>(*this);

   auto frame = ctxt.GetPad()->GetFrame();
   RFrame::RUserRanges ranges;
   if (frame) frame->GetClientRanges(ctxt.GetConnId(), ranges);

   printf("Frame X min %5.3f max %5.3f\n", ranges.GetMin(0), ranges.GetMax(0));
   printf("Frame Y min %5.3f max %5.3f\n", ranges.GetMin(1), ranges.GetMax(1));

   auto himpl = fHistImpl.get();

   if (himpl) {

      int nbinsx = himpl->GetAxis(0).GetNBinsNoOver();
      int nbinsy = himpl->GetAxis(1).GetNBinsNoOver();

      int i1 = 0, i2 = nbinsx, j1 = 0, j2 = nbinsy;

      if (ranges.HasMin(0) && ranges.HasMax(0) && (ranges.GetMin(0) != ranges.GetMax(0))) {
         i1 = himpl->GetAxis(0).FindBin(ranges.GetMin(0));
         i2 = himpl->GetAxis(0).FindBin(ranges.GetMax(0));
         if (i1 <= 1) i1 = 0; else i1--; // include extra left bin
         if ((i2 <= 0) || (i2 >= nbinsx)) i2 = nbinsx; else i2++; // include extra right bin
      }

      if (ranges.HasMin(1) && ranges.HasMax(1) && (ranges.GetMin(1) != ranges.GetMax(1))) {
         j1 = himpl->GetAxis(1).FindBin(ranges.GetMin(1));
         j2 = himpl->GetAxis(1).FindBin(ranges.GetMax(1));
         if (j1 <= 1) j1 = 0; else j1--; // include extra left bin
         if ((j2 <= 0) || (j2 >= nbinsy)) j2 = nbinsy; else j2++; // include extra right bin
      }

      printf("Get indicies %d %d %d %d\n", i1, i2, j1, j2);

      if ((i1>=i2) || (j1>=j2)) {
         printf("FATAL, fallback to full content\n");
         i1 = 0; i2 = nbinsx; j1 = 0; j2 = nbinsy;
      }

      auto &bins = item->GetBinContent();
      bins.resize((i2 - i1) * (j2 - j1));

      double min{0}, minpos{0}, max{0};

      min = max = himpl->GetBinContentAsDouble(1);
      if (max > 0) minpos = max;
      int previndx = -1;

      for (int j = 0; j < nbinsy; ++j)
         for (int i = 0; i < nbinsx; ++i) {
            double val = himpl->GetBinContentAsDouble(j*nbinsx + i + 1);
            if (val < min) min = val; else
            if (val > max) max = val;
            if ((val > 0.) && (val < minpos)) minpos = val;
            if ((i>=i1) && (i<i2) && (j>=j1) && (j<j2)) {
               int indx = (j-j1)*(i2-i1) + (i-i1);
               if ((indx != previndx+1) || (indx < 0) || (indx >= (int) bins.size()))
                  printf("Index mismatch %d prev %d max %d\n", indx, previndx, (int) bins.size());
               else
                  bins[indx] = val;
               previndx = indx;
            }
         }

      item->SetContentMinMax(min, minpos, max);
      item->AddAxis(&himpl->GetAxis(0), i1, i2);
      item->AddAxis(&himpl->GetAxis(1), j1, j2);
   }

   return item;
}
