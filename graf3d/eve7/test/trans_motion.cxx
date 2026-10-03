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
#include <ROOT/REveElement.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveUtil.hxx>

#include <nlohmann/json.hpp>

namespace REX = ROOT::Experimental;

TEST(REveTransMotion, SetMotion)
{
   REX::REveTrans t;
   EXPECT_FALSE(t.HasMotion());

   const double before = REX::REveUtil::ServerTimeMs();
   t.SetMotion({1, 2, 3}, {0, -9.81, 0}, {0, 0, 2}, 3., 0.5);
   const double after = REX::REveUtil::ServerTimeMs();

   ASSERT_TRUE(t.HasMotion());
   const auto &d = *t.GetDeltaTrans();
   EXPECT_DOUBLE_EQ(d.fVel.fX, 1.);
   EXPECT_DOUBLE_EQ(d.fVel.fY, 2.);
   EXPECT_DOUBLE_EQ(d.fVel.fZ, 3.);
   EXPECT_DOUBLE_EQ(d.fAcc.fY, -9.81);
   // The axis is normalised.
   EXPECT_DOUBLE_EQ(d.fSpinAxis.fX, 0.);
   EXPECT_DOUBLE_EQ(d.fSpinAxis.fY, 0.);
   EXPECT_DOUBLE_EQ(d.fSpinAxis.fZ, 1.);
   EXPECT_DOUBLE_EQ(d.fSpinRate, 3.);
   EXPECT_DOUBLE_EQ(d.fMaxDt, 0.5);
   // t0 is stamped when the motion is set.
   EXPECT_GE(d.fMotionT0, before);
   EXPECT_LE(d.fMotionT0, after);

   t.ClearMotion();
   EXPECT_FALSE(t.HasMotion());
   EXPECT_EQ(t.GetDeltaTrans(), nullptr);
}

// A zero axis means no spin, whatever the rate.
TEST(REveTransMotion, ZeroAxisNoSpin)
{
   REX::REveTrans t;
   t.SetMotion({1, 0, 0}, {0, 0, 0}, {0, 0, 0}, 5., 1.);
   ASSERT_TRUE(t.HasMotion());
   EXPECT_DOUBLE_EQ(t.GetDeltaTrans()->fSpinRate, 0.);
   EXPECT_DOUBLE_EQ(t.GetDeltaTrans()->fSpinAxis.Mag(), 1.);

   REX::REveTrans u;
   u.SetMotion({1, 0, 0}, {0, 0, 0}, 1.);
   ASSERT_TRUE(u.HasMotion());
   EXPECT_DOUBLE_EQ(u.GetDeltaTrans()->fSpinRate, 0.);
}

// Copies own their motion state.
TEST(REveTransMotion, CopyIsDeep)
{
   REX::REveTrans a;
   a.SetMotion({1, 0, 0}, {0, 0, 0}, 1.);

   REX::REveTrans b(a);
   ASSERT_TRUE(b.HasMotion());
   EXPECT_NE(b.GetDeltaTrans(), a.GetDeltaTrans());
   b.SetMotion({7, 0, 0}, {0, 0, 0}, 1.);
   EXPECT_DOUBLE_EQ(a.GetDeltaTrans()->fVel.fX, 1.);

   REX::REveTrans c;
   c = a;
   ASSERT_TRUE(c.HasMotion());
   EXPECT_DOUBLE_EQ(c.GetDeltaTrans()->fVel.fX, 1.);

   // Assigning a transformation without motion clears it.
   REX::REveTrans still;
   c = still;
   EXPECT_FALSE(c.HasMotion());
}

TEST(REveTransMotion, TransJson)
{
   REX::REveManager::Create();
   auto el = new REX::REveElement("mover");

   nlohmann::json j0;
   el->WriteTransJson(j0);
   EXPECT_FALSE(j0.contains("matrix"));
   EXPECT_TRUE(j0["mot"].is_null());

   el->RefMainTrans().SetPos(4, 5, 6);
   el->RefMainTrans().SetMotion({1, 2, 3}, {0, -1, 0}, {2, 0, 0}, 0.5, 0.25);

   nlohmann::json j;
   el->WriteTransJson(j);
   ASSERT_EQ(j["matrix"].size(), 16u);
   EXPECT_DOUBLE_EQ(j["matrix"][12].get<double>(), 4.);
   auto &m = j["mot"];
   ASSERT_TRUE(m.is_object());
   EXPECT_EQ(m["vel"], nlohmann::json({1., 2., 3.}));
   EXPECT_EQ(m["acc"], nlohmann::json({0., -1., 0.}));
   EXPECT_EQ(m["axis"], nlohmann::json({1., 0., 0.}));
   EXPECT_DOUBLE_EQ(m["rate"].get<double>(), 0.5);
   EXPECT_DOUBLE_EQ(m["max_dt"].get<double>(), 0.25);
   EXPECT_DOUBLE_EQ(m["t0"].get<double>(), el->RefMainTrans().GetDeltaTrans()->fMotionT0);

   // A stopped element sends an explicit null, so the client drops its trajectory.
   el->RefMainTrans().ClearMotion();
   nlohmann::json j2;
   el->WriteTransJson(j2);
   EXPECT_TRUE(j2.contains("mot"));
   EXPECT_TRUE(j2["mot"].is_null());

   delete el;
}
