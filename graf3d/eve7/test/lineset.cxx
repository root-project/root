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
#include <ROOT/REveRenderData.hxx>

#include <nlohmann/json.hpp>

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

   Int_t ntotm = 0;
   for (Int_t i = 0; i<nlines; i++) {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for (Int_t m = 0; m < nm; m++) ls->AddMarker(i, r.Rndm());
      ntotm += nm;
   }

   ls->SetMarkerSize(1.5);
   ls->SetMarkerStyle(4);
   eveMng->GetEventScene()->AddElement(ls);

   EXPECT_EQ(ls->GetLinePlex().Size(), nlines);
   EXPECT_EQ(ls->GetMarkerPlex().Size(), ntotm);

   // Two vertices per line and one per marker; one index per line and per
   // marker, holding the line id.
   nlohmann::json j;
   Int_t bin = ls->WriteCoreJson(j, 0);
   EXPECT_EQ(j["fLinePlexSize"], nlines);
   EXPECT_EQ(j["fMarkerPlexSize"], ntotm);
   EXPECT_EQ(j["render_data"]["rnr_func"], "makeStraightLineSet");
   EXPECT_EQ(j["render_data"]["vert_size"], 3 * (2 * nlines + ntotm));
   EXPECT_EQ(j["render_data"]["index_size"], nlines + ntotm);
   EXPECT_EQ(bin, ls->GetRenderData()->GetBinarySize());
}
