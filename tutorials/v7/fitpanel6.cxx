/// \file
/// \ingroup tutorial_v7
///
/// \macro_code
///
/// \date 2019-04-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
/// \author Sergey Linev <S.Linev@gsi.de>
/// \author Iliana Bessou <Iliana.Bessou@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFitPanel6.hxx>
#include "TH1.h"

void fitpanel6()
{
   auto panel = new ROOT::Experimental::RFitPanel6();

   TH1F *hpx = new TH1F("test","This is test histogram",100,-4,4);
   hpx->FillRandom("gaus", 10000);
   hpx->Draw();

   // TFile::Open("hsimple.root");
   // gFile->Get("hpx");
   // gFile->Get("hpxpx");
   // gFile->Get("hprof");

   panel->AssignHistogram(hpx);

   panel->Show();
}

