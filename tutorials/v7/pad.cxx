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

#include "Rtypes.h"

R__LOAD_LIBRARY(libROOTGpadv7);

#include "ROOT/RCanvas.hxx"
#include "ROOT/RFrame.hxx"
#include "ROOT/Rline.hxx"

void pad()
{
   using namespace ROOT;
   using namespace ROOT::Experimental;

   auto canvas = Experimental::RCanvas::Create("what to do with a pad!");
   auto pads   = canvas->Divide(3, 3);
   auto &pad12 = pads[1][2];
   pad12->SetAllAxisBounds({{50., 250.}, {-1., 1.}});
   // Please fix Rline such that {x,y} are TPadPos!
   //pad12->Draw(Experimental::Rline({100._user, 0.5_normal}, {200._user, 0.5_normal}));

   canvas->Show();
}
