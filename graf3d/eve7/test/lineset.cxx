// @(#)root/eve7:$Id$
// Author: Sergey Linev, 2019-02-26

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "gtest/gtest.h"

#include "TRandom.h"

#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveStraightLineSet.hxx>

// Test creation of LineSet
TEST(REveManager, LinesSet) {
   namespace REX = ROOT::Experimental;

   Int_t nlines = 40, nmarkers = 4;

   auto eveMng = REX::REveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   auto ls = new REX::REveStraightLineSet();
   ls->SetMainColor(kBlue);
   ls->SetMarkerColor(kRed);

   for (Int_t i = 0; i<nlines; i++) {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for (Int_t m = 0; m < nm; m++) ls->AddMarker(i, r.Rndm());
   }

   ls->SetMarkerSize(1.5);
   ls->SetMarkerStyle(4);
   eveMng->GetEventScene()->AddElement(ls);
}
