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
   auto line       = std::make_shared<Experimental::TLine>(0. ,0. ,1. , 1.);
   canvas->Draw(line);
      
      
   // Register the line with ROOT: now it lives even after draw() ends.
   // Experimental::TDirectory::Heap().Add("line", line);

   canvas->Show();

   // TFile *f = TFile::Open("canv7.root", "recreate");
   // f->WriteObject(canvas.get(), "canv_line");
   // delete f;
}
