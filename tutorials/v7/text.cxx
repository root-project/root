/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2017-10-17
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
#include "ROOT/TText.hxx"
#include "ROOT/TDirectory.hxx"

void text()
{
   using namespace ROOT;

   // Create a canvas to be displayed.
   auto canvas = Experimental::TCanvas::Create("Canvas Title");

   for (int i=0; i<=360; i+=10) {
      auto text       = std::make_shared<Experimental::TText>(.5,.4, "____  Hello World");
      canvas->Draw(text);
      
      auto col = Experimental::TColor(0.0015*i, 0.0025*i ,0.003*i);
      text->GetOptions().SetTextColor(col);
      text->GetOptions().SetTextSize(10+i/10);
      text->GetOptions().SetTextAngle(i);
   }

   // Register the text with ROOT: now it lives even after draw() ends.
   // Experimental::TDirectory::Heap().Add("text", text);

   canvas->Show();

   // TFile *f = TFile::Open("canv7.root", "recreate");
   // f->WriteObject(canvas.get(), "canv_text");
   // delete f;
}
