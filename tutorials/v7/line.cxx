/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2018-03-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
/// \author Olivier couet <Olivier.Couet@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

R__LOAD_LIBRARY(libGpad);

// #include "ROOT/TFile.hxx"
#include "ROOT/TCanvas.hxx"
#include "ROOT/TColor.hxx"
#include "ROOT/TLine.hxx"
#include "ROOT/TDirectory.hxx"

void line()
{
   using namespace ROOT;

   // Create a canvas to be displayed.
   auto canvas = Experimental::TCanvas::Create("Canvas Title");
 


   for (double i = 0; i < 360; i+=1){
        double ang = i * TMath::Pi() / 180;
        double x = 0.3*TMath::Cos(ang) + 0.5;
        double y = 0.3*TMath::Sin(ang) + 0.5;

        auto line = std::make_shared<Experimental::TLine>(0.5 ,0.5 ,x, y);

        auto col = Experimental::TColor(0.0025*i, 0, 0);
        line->GetOptions().SetLineColor(col);
        line->GetOptions().SetLineWidth(1);
        //line->GetOptions().SetLineColorAlpha(0.45);
        canvas->Draw(line);

    }

    auto line  = std::make_shared<Experimental::TLine>(0. ,0. ,1. ,1.);
    auto line1 = std::make_shared<Experimental::TLine>(0.1 ,0.1 ,0.9 ,0.1);
    auto line2 = std::make_shared<Experimental::TLine>(0.9 ,0.1 ,0.9 ,0.9);
    auto line3 = std::make_shared<Experimental::TLine>(0.9 ,0.9 ,0.1 ,0.9);
    auto line4 = std::make_shared<Experimental::TLine>(0.1 ,0.1 ,0.1 ,0.9);
    auto line0 = std::make_shared<Experimental::TLine>(0. ,1. ,1. ,0.);


    canvas->Draw(line);
    canvas->Draw(line1);
    canvas->Draw(line2);
    canvas->Draw(line3);
    canvas->Draw(line4);
    canvas->Draw(line0);



      
   // Register the line with ROOT: now it lives even after draw() ends.
   // Experimental::TDirectory::Heap().Add("line", line);

   canvas->Show();

   // TFile *f = TFile::Open("canv7.root", "recreate");
   // f->WriteObject(canvas.get(), "canv_line");
   // delete f;
}
