// @(#)root/eve7:$Id$

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "gtest/gtest.h"

#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveProjectionAxis.hxx>
#include <ROOT/REveSMorph.hxx>
#include <ROOT/REveLogo.hxx>

#include <nlohmann/json.hpp>

#include <string>

namespace REX = ROOT::Experimental;

TEST(REveSMorph, Clamping)
{
   REX::REveManager::Create();
   auto m = new REX::REveSMorph("morph");

   m->SetTLevel(1);
   EXPECT_EQ(m->GetTLevel(), 2);
   m->SetTLevel(500);
   EXPECT_EQ(m->GetTLevel(), 200);
   m->SetPLevel(2);
   EXPECT_EQ(m->GetPLevel(), 3);

   m->SetTx(5.f);
   EXPECT_FLOAT_EQ(m->GetTx(), 2.f);
   m->SetCx(-5.f);
   EXPECT_FLOAT_EQ(m->GetCx(), -2.f);

   m->SetThetaMin(-1.f);
   EXPECT_FLOAT_EQ(m->GetThetaMin(), 0.f);
   m->SetPhiRange(3.f);
   EXPECT_FLOAT_EQ(m->GetPhiRange(), 1.f);

   delete m;
}

TEST(REveSMorph, Json)
{
   REX::REveManager::Create();
   auto m = new REX::REveSMorph("morph");
   m->SetTLevel(12);
   m->SetPLevel(16);
   m->SetTx(0.5f);
   m->SetThetaMax(0.5f);
   m->SetTexture("checker_8.png");

   nlohmann::json j;
   int bin = m->WriteCoreJson(j, 0);
   EXPECT_GT(bin, 0);
   EXPECT_EQ(j["render_data"]["rnr_func"], "makeSMorph");
   EXPECT_EQ(j["fTLevel"], 12);
   EXPECT_EQ(j["fPLevel"], 16);
   EXPECT_FLOAT_EQ(j["fTx"].get<float>(), 0.5f);
   EXPECT_FLOAT_EQ(j["fThetaMax"].get<float>(), 0.5f);
   EXPECT_EQ(j["fTexture"], "checker_8.png");

   delete m;
}

// Ticks for a known extent, without distortion: in kValue mode the label of a
// major tick is its position, and 0 is labelled "0".
TEST(REveProjectionAxis, Ticks)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("axis_src");
   auto dst = eve->SpawnNewScene("axis_dst");

   auto ps = new REX::REvePointSet("extent");
   ps->SetNextPoint(-100, -100, -100);
   ps->SetNextPoint(100, 100, 100);
   src->AddElement(ps);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   mng->ImportElements(ps, dst);

   auto axis = new REX::REveProjectionAxis(mng);
   dst->AddElement(axis);
   axis->UpdateTicks();

   nlohmann::json j;
   axis->WriteCoreJson(j, -1);

   for (const char *k : {"H", "V"}) {
      std::string key(k);
      auto &pos = j["fTickPos" + key];
      auto &lab = j["fTickLab" + key];
      auto &maj = j["fTickMaj" + key];
      ASSERT_FALSE(pos.empty()) << key;
      ASSERT_EQ(pos.size(), lab.size());
      ASSERT_EQ(pos.size(), maj.size());

      bool has_zero = false;
      for (size_t i = 0; i < pos.size(); ++i) {
         if (i > 0)
            EXPECT_LT(pos[i - 1].get<float>(), pos[i].get<float>()) << key << " not sorted at " << i;
         std::string l = lab[i];
         if (maj[i].get<bool>()) {
            EXPECT_NEAR(std::stod(l), pos[i].get<float>(), 1e-3) << key << " label " << l;
            has_zero |= (l == "0");
         } else {
            EXPECT_TRUE(l.empty()) << key << " minor tick with a label";
         }
      }
      EXPECT_TRUE(has_zero) << key;
   }
}

TEST(REveLogo, IsRemote)
{
   EXPECT_TRUE(REX::REveLogo::IsRemote("http://root.cern/logo.png"));
   EXPECT_TRUE(REX::REveLogo::IsRemote("https://root.cern/logo.png"));
   EXPECT_TRUE(REX::REveLogo::IsRemote("//root.cern/logo.png"));
   EXPECT_FALSE(REX::REveLogo::IsRemote("logo.png"));
   EXPECT_FALSE(REX::REveLogo::IsRemote("/abs/logo.png"));
   EXPECT_FALSE(REX::REveLogo::IsRemote("httpish.png"));
}
