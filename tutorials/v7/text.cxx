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

   auto text       = std::make_shared<ROOT::Experimental::TText>(.5,.8, "Hello World");
   auto drawn_text = canvas->Draw(text);

   drawn_text->SetTextColor(Experimental::TColor::kRed);

   text->GetOptions().SetTextSize(40);

   cout << endl;
   cout << ">>>>> Text position : "<< text->GetX() << " " << text->GetY() << endl;
   cout << ">>>>> Text string :   "<< text->GetText() << endl;
   cout << ">>>>> Text size  :    "<< (int)text->GetOptions().GetTextSize() << endl;
   cout << endl;

   // Register the text with ROOT: now it lives even after draw() ends.
   // Experimental::TDirectory::Heap().Add("text", text);

   canvas->Show();

   // TFile *f = TFile::Open("canv7.root", "recreate");
   // f->WriteObject(canvas.get(), "canv_text");
   // delete f;
}
