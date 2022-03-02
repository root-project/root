/// \file
/// \ingroup tutorial_rcanvas
///
/// This macro generates really large RH1D histogram, fills it with predefined pattern and
/// draw it in a RCanvas, using Optmize() drawing mode
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2020-07-02
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
#include "TMath.h"
#include "TString.h"


// macro must be here while cling is not capable to load
// library automatically for outlined function see ROOT-10336
R__LOAD_LIBRARY(libROOTHistDraw)

using namespace ROOT::Experimental;

void rh1_large()
{
   const int nbins = 5000000;

   // Create the histogram.
   RAxisConfig xaxis("x", nbins, 0., nbins);
   auto pHist = std::make_shared<RH1D>(xaxis);

   for(int i=0;i<nbins;++i)
      pHist->Fill(1.*i, 1000.*(2+TMath::Sin(100.*i/nbins)));

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Drawing large RH1");

   auto frame = canvas->AddFrame();

   frame->gridX = true;
   frame->gridY = true;
   frame->x.zoomMin = nbins*0.2;
   frame->x.zoomMax = nbins*0.8;

   canvas->Draw<RFrameTitle>(TString::Format("Large RH1D histogram with %d bins",nbins).Data());

   auto draw = canvas->Draw(pHist);

   draw->line.color = RColor::kLime;
   // draw->fill.color = RColor::kLime;
   // draw->fill.style = RAttrFill::kSolid;
   // draw->Line(); // configure line draw option
   // draw->Bar(); // configure bar draw option
   // draw->Error(3); // configure error drawing
   draw->Hist();  // configure hist draw option, default

   draw->optimize = true; // enable draw optimization, reduced data set will be send to clients

   auto stat = canvas->Draw<RHist1StatBox>(pHist, "hist");
   stat->fill.color = RColor::kBlue;
   stat->fill.style = RAttrFill::kSolid;

   canvas->SetSize(1000, 700);
   canvas->Show();
}
