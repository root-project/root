/// \file RNTupleDrawVisitor.cxx
/// \ingroup NTuple
/// \author Sergey Linev <S.Linev@gsi.de>, Jakob Blomer <jblomer@cern.ch>
/// \date 2025-07-24

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleDrawVisitor.hxx>

#include <TH1F.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

void ROOT::Internal::RNTupleDrawVisitor::TestHistBuffer()
{
   std::size_t len = fHist->GetBufferLength();
   auto buf = fHist->GetBuffer();

   if (!buf || (len < 5))
      return;

   double min = buf[1];
   double max = buf[1];
   bool is_integer = true;

   for (std::size_t n = 0; n < len; ++n) {
      double v = buf[2 + 2 * n];
      max = std::max(max, v);
      min = std::min(min, v);
      double _;
      if (std::abs(std::modf(v, &_)) > 1e-5) {
         is_integer = false;
         break;
      }
   }

   // special case when only integer values in short range - better binning
   if (is_integer && (max - min < 100)) {
      max += 2;
      if (min > 1)
         min -= 2;
      int npoints = std::nearbyint(max - min);
      std::unique_ptr<TH1> h1 = std::make_unique<TH1F>(fHist->GetName(), fHist->GetTitle(), npoints, min, max);
      h1->SetDirectory(nullptr);
      for (size_t n = 0; n < len; ++n)
         h1->Fill(buf[2 + 2 * n], buf[1 + 2 * n]);
      std::swap(fHist, h1);
   }
}
