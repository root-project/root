/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2015-03-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Axel Naumann <axel@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHist.hxx"
#include "ROOT/RFit.hxx"
#include "ROOT/RFile.hxx"

void simple()
{
   using namespace ROOT;

   // Create a 2D histogram with an X axis with equidistant bins, and a y axis
   // with irregular binning.
   Experimental::RAxisConfig xAxis(100, 0., 1.);
   Experimental::RAxisConfig yAxis({0., 1., 2., 3., 10.});
   Experimental::RH2D histFromVars(xAxis, yAxis);

   // Or the short in-place version:
   // Create a 2D histogram with an X axis with equidistant bins, and a y axis
   // with irregular binning.
   Experimental::RH2D hist({100, 0., 1.}, {{0., 1., 2., 3., 10.}});

   // Fill weight 1. at the coordinate 0.01, 1.02.
   hist.Fill({0.01, 1.02});

   // Fit the histogram.
   Experimental::TFunction<2> func([](const std::array<double, 2> &x, const std::span<double> par) {
      return par[0] * x[0] * x[0] + (par[1] - x[1]) * x[1];
   });

   auto fitResult = Experimental::FitTo(hist, func, {{0., 1.}});

   auto file = Experimental::RFile::Recreate("hist.root");
   file->Write("TheHist", hist);
}
