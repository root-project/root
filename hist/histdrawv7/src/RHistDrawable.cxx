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

#include <algorithm> // for std::min

using namespace ROOT::Experimental;


std::unique_ptr<RDisplayItem> RHist1Drawable::CreateHistDisplay(const RDisplayContext &ctxt)
{
   auto item = std::make_unique<RHistDisplayItem>(*this);

   auto frame = ctxt.GetPad()->GetFrame();
   RFrame::RUserRanges ranges;
   if (frame) frame->GetClientRanges(ctxt.GetConnId(), ranges);

   auto himpl = fHistImpl.get();

   if (himpl) {

      int nbinsx = himpl->GetAxis(0).GetNBinsNoOver();

      int i1 = 0, i2 = nbinsx, stepi = 1;

      if (ranges.HasMin(0) && ranges.HasMax(0) && (ranges.GetMin(0) != ranges.GetMax(0))) {
         i1 = himpl->GetAxis(0).FindBin(ranges.GetMin(0));
         i2 = himpl->GetAxis(0).FindBin(ranges.GetMax(0));
         if (i1 <= 1) i1 = 0; else i1--; // include extra left bin
         if ((i2 <= 0) || (i2 >= nbinsx)) i2 = nbinsx; else i2++; // include extra right bin
      }

      if (i1 >= i2) {
         i1 = 0; i2 = nbinsx;
      }

      // make approx 200 bins visible in each direction
      static const int NumVisibleBins = 5000;

      bool needrebin = false;

      if (i2 - i1 > NumVisibleBins) {
         stepi = (i2 - i1) / NumVisibleBins;
         if (stepi < 2) stepi = 2;
         i1 = (i1 / stepi) * stepi;
         if (i2 % stepi > 0) i2 = (i2/stepi + 1) * stepi;
         needrebin = true;
      }

      printf("RH1 Get indicies X: %d %d %d\n", i1, i2, stepi);

      // be aware, right indicies i2 or j2 can be larger then axes index range.
      // In this case axis rebin is special

      auto &bins = item->GetBinContent();
      bins.resize((i2 - i1) / stepi);

      double min{0}, minpos{0}, max{0};

      min = max = himpl->GetBinContentAsDouble(1);
      if (max > 0) minpos = max;

      /// found histogram min/max values
      for (int i = 0; i < nbinsx; ++i) {
         double val = himpl->GetBinContentAsDouble(i + 1);
         if (val < min) min = val; else
         if (val > max) max = val;
         if ((val > 0.) && (val < minpos)) minpos = val;
         if (!needrebin && (i>=i1) && (i<i2)) {
            int indx = (i-i1);
            bins[indx] = val;
         }
      }

      // provide simple rebin with average values
      // TODO: provide methods in histogram classes
      if (needrebin)
         for (int i = i1; i < i2; i += stepi) {
            double sum = 0.;
            int ir = std::min(i+stepi, nbinsx);
            for(int ii = i; ii < ir; ++ii)
               sum += himpl->GetBinContentAsDouble(ii + 1);
            int indx = (i-i1)/stepi;
            bins[indx] = sum/(ir-i);
         }

      item->SetContentMinMax(min, minpos, max);
      item->AddAxis(&himpl->GetAxis(0), i1, i2, stepi);
   }

   return item;
}


std::unique_ptr<RDisplayItem> RHist2Drawable::CreateHistDisplay(const RDisplayContext &ctxt)
{
   auto item = std::make_unique<RHistDisplayItem>(*this);

   auto frame = ctxt.GetPad()->GetFrame();
   RFrame::RUserRanges ranges;
   if (frame) frame->GetClientRanges(ctxt.GetConnId(), ranges);

   auto himpl = fHistImpl.get();

   if (himpl) {

      int nbinsx = himpl->GetAxis(0).GetNBinsNoOver();
      int nbinsy = himpl->GetAxis(1).GetNBinsNoOver();

      int i1 = 0, i2 = nbinsx, j1 = 0, j2 = nbinsy, stepi = 1, stepj = 1;

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

      if ((i1 >= i2) || (j1 >= j2)) {
         printf("FATAL, fallback to full content\n");
         i1 = 0; i2 = nbinsx; j1 = 0; j2 = nbinsy;
      }

      // make approx 200 bins visible in each direction
      static const int NumVisibleBins = 200;

      bool needrebin = false;

      if (i2 - i1 > NumVisibleBins) {
         stepi = (i2 - i1) / NumVisibleBins;
         if (stepi < 2) stepi = 2;
         i1 = (i1 / stepi) * stepi;
         if (i2 % stepi > 0) i2 = (i2/stepi + 1) * stepi;
         needrebin = true;
      }

      if (j2 - j1 > NumVisibleBins) {
         stepj = (j2 - j1) / NumVisibleBins;
         if (stepj < 2) stepj = 2;
         j1 = (j1 / stepj) * stepj;
         if (j2 % stepj > 0) j2 = (j2/stepj + 1) * stepj;
         needrebin = true;
      }

      // be aware, right indicies i2 or j2 can be larger then axes index range.
      // In this case axis rebin is special

      auto &bins = item->GetBinContent();
      bins.resize((i2 - i1) / stepi * (j2 - j1) / stepj);

      double min{0}, minpos{0}, max{0};

      min = max = himpl->GetBinContentAsDouble(1);
      if (max > 0) minpos = max;

      /// found histogram min/max values
      for (int j = 0; j < nbinsy; ++j)
         for (int i = 0; i < nbinsx; ++i) {
            double val = himpl->GetBinContentAsDouble(j*nbinsx + i + 1);
            if (val < min) min = val; else
            if (val > max) max = val;
            if ((val > 0.) && (val < minpos)) minpos = val;
            if (!needrebin && (i>=i1) && (i<i2) && (j>=j1) && (j<j2)) {
               int indx = (j-j1)*(i2-i1) + (i-i1);
               bins[indx] = val;
            }
         }

      // provide simple rebin with average values
      // TODO: provide methods in histogram classes
      if (needrebin)
         for (int j = j1; j < j2; j += stepj) {
            int jr = std::min(j+stepj, nbinsy);
            for (int i = i1; i < i2; i += stepi) {
               double sum = 0.;
               int ir = std::min(i+stepi, nbinsx);
               int cnt = 0;
               for(int jj = j; jj < jr; ++jj)
                  for(int ii = i; ii < ir; ++ii) {
                     sum += himpl->GetBinContentAsDouble(jj*nbinsx + ii + 1);
                     cnt++;
                  }

               int indx = (j-j1)/stepj * (i2-i1)/stepi + (i-i1)/stepi;
               bins[indx] = sum/cnt;
            }
         }

      item->SetContentMinMax(min, minpos, max);
      item->AddAxis(&himpl->GetAxis(0), i1, i2, stepi);
      item->AddAxis(&himpl->GetAxis(1), j1, j2, stepj);
   }

   return item;
}


std::unique_ptr<RDisplayItem> RHist3Drawable::CreateHistDisplay(const RDisplayContext & /*ctxt*/)
{
   auto item = std::make_unique<RHistDisplayItem>(*this);

   return item;

}
