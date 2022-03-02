/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro generates really large RH2D histogram, fills it with predefined pattern and
/// draw it in a RCanvas, using Optmize() drawing mode
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2020-06-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RHistDrawable.hxx"
#include "ROOT/RCanvas.hxx"
#include "ROOT/RFrameTitle.hxx"
#include "ROOT/RHistStatBox.hxx"
#include "ROOT/RFrame.hxx"
#include "TString.h"

// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void rh3_large()
{
   const int nbins = 200;

   // Create the histogram.
   RAxisConfig xaxis("x", nbins, 0., nbins);
   RAxisConfig yaxis("y", nbins, 0., nbins);
   RAxisConfig zaxis("z", nbins, 0., nbins);
   auto pHist = std::make_shared<RH3D>(xaxis, yaxis, zaxis);

   for(int i=0;i<nbins;++i)
      for(int j=0;j<nbins;++j)
         for(int k=0;k<nbins;++k)
            pHist->Fill({1.*i,1.*j,1.*k}, i+j+k);

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Large 200x200x200 RH3 drawing");

   auto frame = canvas->AddFrame();

   // should we made special style for frame with palette?
   // frame->margins.right = 0.2_normal;

   frame->x.zoomMin = nbins*0.1;
   frame->x.zoomMax = nbins*0.9;
   frame->y.zoomMin = nbins*0.1;
   frame->y.zoomMax = nbins*0.9;
   frame->z.zoomMin = nbins*0.1;
   frame->z.zoomMax = nbins*0.9;

   canvas->Draw<RFrameTitle>(TString::Format("Large RH3D histogram with %d x %d x %d bins",nbins,nbins,nbins).Data());

   auto draw = canvas->Draw(pHist);

   draw->line.color = RColor::kLime;
   // draw->Box(); // configure box draw option (default)
   // draw->Sphere(); // configure sphere draw option
   draw->Scatter(); // configure scatter draw option
   // draw->Color(); // configure color draw option

   draw->optimize = true; // enable draw optimization, reduced data set will be send to clients

   // auto stat = canvas->Draw<RHist2StatBox>(pHist, "hist");
   // stat->fill.color = RColor::kBlue;
   // stat->fill.style = RAttrFill::kSolid;

   canvas->SetSize(1000, 700);
   canvas->Show();
}
