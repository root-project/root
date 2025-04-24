/// \file
/// \ingroup tutorial_webgui
///
/// \macro_code
///
/// \date 2019-04-11
/// \warning This is part of the experimental API, which might change in the future. Feedback is welcome!
/// \authors Sergey Linev <S.Linev@gsi.de>, Iliana Betsou <Iliana.Betsou@cern.ch>

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RFitPanel.hxx>
#include "TH1.h"
#include "TFile.h"

void fitpanel6()
{
   TFile::Open("hsimple.root");
   if (gFile) {
      gFile->Get("hpx");
      gFile->Get("hpxpy");
      gFile->Get("hprof");
   }

   // create panel
   auto panel = std::make_shared<ROOT::Experimental::RFitPanel>("FitPanel");

   TH1F *test = new TH1F("test", "This is test histogram", 100, -4, 4);
   test->FillRandom("gaus", 10000);

   panel->AssignHistogram(test);

   panel->Show();

   panel->ClearOnClose(panel);
}
