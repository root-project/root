/// \file
/// \ingroup tutorial_rcanvas
///
/// Macro illustrating how to superimpose two histograms
/// with different scales on the RCanvas. It shows exactly same data
/// as in hist/twoscales.C macro, but with fully interactive graphics
///
/// \macro_image (rcanvas_js)
/// \macro_code
///
/// \date 2021-07-05
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <s.linev@gsi.de>

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCanvas.hxx"
#include "ROOT/TObjectDrawable.hxx"
#include "TH1.h"
#include "TRandom.h"

using namespace ROOT::Experimental;

void th1_twoscales()
{
   //create/fill draw h1
   auto h1 = std::make_shared<TH1F>("h1","Example histogram",100,-3,3);
   h1->SetDirectory(nullptr);
   h1->SetStats(kFALSE);
   for (int i=0;i<10000;i++)
      h1->Fill(gRandom->Gaus(0,1));

   //create hint1 filled with the bins integral of h1
   auto hint1 = std::make_shared<TH1F>("hint1","h1 bins integral",100,-3,3);
   hint1->SetDirectory(nullptr);
   hint1->SetStats(kFALSE);
   Float_t sum = 0;
   for (int i=1;i<=100;i++) {
      sum += h1->GetBinContent(i);
      hint1->SetBinContent(i,sum);
   }
   hint1->SetLineColor(kRed);
   hint1->GetYaxis()->SetAxisColor(kRed);
   hint1->GetYaxis()->SetLabelColor(kRed);

   // Create a canvas to be displayed.
   auto canvas = RCanvas::Create("Two TH1 with two independent Y scales");

   // just draw histogram on RCanvas
   canvas->Draw<TObjectDrawable>(h1, "");

   // add second histogram and specify Y+ draw option
   canvas->Draw<TObjectDrawable>(hint1, "same,Y+");

   // new window in web browser should popup
   canvas->Show();

   // create PNG file
   // canvas->SaveAs("th1_twoscales.png");
}
