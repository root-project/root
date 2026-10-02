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
#include <ROOT/REveProjections.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveProjectionManager.hxx>

namespace REX = ROOT::Experimental;

TEST(REveProjection, RPhiWithoutDistortion)
{
   REX::REveRPhiProjection p;
   Float_t x = 3, y = 4, z = 5;
   p.ProjectPoint(x, y, z, 0.5f, REX::REveProjection::kPP_Full);
   EXPECT_FLOAT_EQ(x, 3.f);
   EXPECT_FLOAT_EQ(y, 4.f);
   EXPECT_FLOAT_EQ(z, 0.5f); // the depth
}

// RhoZ maps (x, y, z) to (z, +-rho); the sign is the side of the y = 0 plane.
TEST(REveProjection, RhoZSign)
{
   REX::REveRhoZProjection p;
   Float_t x = 3, y = 4, z = 5;
   p.ProjectPoint(x, y, z, 0.f, REX::REveProjection::kPP_Full);
   EXPECT_FLOAT_EQ(x, 5.f);
   EXPECT_FLOAT_EQ(y, 5.f);

   x = 3, y = -4, z = 5;
   p.ProjectPoint(x, y, z, 0.f, REX::REveProjection::kPP_Full);
   EXPECT_FLOAT_EQ(x, 5.f);
   EXPECT_FLOAT_EQ(y, -5.f);
}

// Distortion magnifies radii below fFixR and keeps fFixR itself in place.
TEST(REveProjection, DistortionFixedRadius)
{
   REX::REveRPhiProjection p;
   p.SetDistortion(0.002f);
   const Float_t fix = p.GetFixR();

   Float_t x = fix, y = 0, z = 0;
   p.ProjectPoint(x, y, z, 0.f, REX::REveProjection::kPP_Full);
   EXPECT_NEAR(x, fix, 1e-3 * fix);

   x = 0.1f * fix, y = 0, z = 0;
   p.ProjectPoint(x, y, z, 0.f, REX::REveProjection::kPP_Full);
   EXPECT_GT(x, 0.1f * fix);
   EXPECT_LT(x, fix);
}

// GetValForScreenPos() inverts GetScreenVal(); REveProjectionAxis relies on it
// for the labels of a distorted projection.
TEST(REveProjection, ScreenValRoundTrip)
{
   REX::REveRPhiProjection rphi;
   REX::REveRhoZProjection rhoz;
   for (REX::REveProjection *p : {(REX::REveProjection *)&rphi, (REX::REveProjection *)&rhoz}) {
      for (Float_t d : {0.f, 0.001f}) {
         p->SetDistortion(d);
         for (int ax = 0; ax < 2; ++ax) {
            for (Float_t v : {-250.f, -40.f, 40.f, 250.f}) {
               Float_t sv = p->GetScreenVal(ax, v);
               EXPECT_NEAR(p->GetValForScreenPos(ax, sv), v, 0.05f)
                  << p->GetName() << " d=" << d << " ax=" << ax << " v=" << v;
            }
         }
      }
   }
}

TEST(REveProjection, ImportAndUpdate)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("proj_src");
   auto dst = eve->SpawnNewScene("proj_dst");

   auto ps = new REX::REvePointSet("model");
   ps->SetNextPoint(3, 4, 5);
   ps->SetNextPoint(-6, 8, -1);
   src->AddElement(ps);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RhoZ);
   auto pr = dynamic_cast<REX::REvePointSet *>(mng->ImportElements(ps, dst));
   ASSERT_NE(pr, nullptr);
   ASSERT_EQ(pr->GetSize(), 2);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fX, 5.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fY, 5.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(1).fX, -1.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(1).fY, 10.f);

   ps->SetPoint(0, 0, -3, 4);
   auto prj = dynamic_cast<REX::REveProjected *>(pr);
   ASSERT_NE(prj, nullptr);
   prj->UpdateProjection();
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fX, 4.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fY, -3.f);
}
