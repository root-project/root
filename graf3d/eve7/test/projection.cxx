// @(#)root/eve7:$Id$

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "gtest/gtest.h"

#include <ROOT/TestSupport.hxx>

#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveClient.hxx>
#include <ROOT/REveCompound.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveUtil.hxx>
#include <ROOT/REveJetCone.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveLine.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveStraightLineSet.hxx>
#include <ROOT/REveProjections.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveTrans.hxx>

#include "TGeoBBox.h"

#include <algorithm>
#include <vector>

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

// ProjectPointdv() may project in place, as REveTrackProjected does for path
// marks; a rotation must not read coordinates it has already overwritten.
TEST(REveProjection, ProjectPointdvInPlace)
{
   REX::REveRPhiProjection p;
   REX::REveTrans t;
   t.RotateLF(1, 2, 0.5);
   t.SetPos(10, 20, 0);

   Double_t in[3] = {3, 4, 0}, out[3];
   p.ProjectPointdv(&t, in, out, 0.f);
   p.ProjectPointdv(&t, in, in, 0.f);
   for (int i = 0; i < 3; ++i)
      EXPECT_DOUBLE_EQ(in[i], out[i]) << "coordinate " << i;
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

// A depth change moves the projected vertices, so it stamps kCBObjProps and
// the client gets the new geometry and box. A REveLine also covers the import
// of a line, whose depth is set before it has points.
TEST(REveProjection, DepthStampsObjProps)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("depth_src");
   auto dst = eve->SpawnNewScene("depth_dst");
   auto win = eve->GetWebWindow();
   dst->AddSubscriber(std::make_unique<REX::REveClient>(999999u, win));

   auto line = new REX::REveLine("model");
   line->SetNextPoint(3, 4, 5);
   line->SetNextPoint(-6, 8, -1);
   src->AddElement(line);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   auto pr = dynamic_cast<REX::REveLine *>(mng->ImportElements(line, dst));
   ASSERT_NE(pr, nullptr);
   ASSERT_EQ(pr->GetSize(), 2);

   eve->BeginChange();
   dynamic_cast<REX::REveProjected *>(pr)->SetDepth(-7);
   EXPECT_TRUE(pr->GetChangeBits() & REX::REveElement::kCBObjProps);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fZ, -7.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(1).fZ, -7.f);
   ASSERT_EQ(pr->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(pr->GetBBoxMin().fZ, -7.f);
   EXPECT_FLOAT_EQ(pr->GetBBoxMax().fZ, -7.f);
   eve->EndChange();
}

// Projected vertices are in the frame of the manager, so import warns when
// they are placed under an element whose matrices differ from the manager's.
// The identity that REveStraightLineSet's constructor sets does not warn; an
// unexpected warning fails the test.
TEST(REveProjection, ImportWarnsAboutTransformation)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("trans_src");
   auto dst = eve->SpawnNewScene("trans_dst");

   auto ls = new REX::REveStraightLineSet("model");
   ls->AddLine(0, 0, 0, 10, 10, 10);
   src->AddElement(ls);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   ASSERT_NE(mng->ImportElements(ls, dst), nullptr);

   auto holder = new REX::REveElement("holder");
   holder->RefMainTrans().SetPos(100, 0, 0);
   dst->AddElement(holder);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "REveProjectionManager::ImportElements", "'holder' places projected elements", false);
   ASSERT_NE(mng->ImportElements(ls, holder), nullptr);
}

// A manager with a matrix projects in its own frame and the client draws its
// children with the same matrix, so importing into it does not warn. Importing
// into an element outside its chain, without that matrix, does.
TEST(REveProjection, ImportIntoTransformedManager)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("tm_src");
   auto dst = eve->SpawnNewScene("tm_dst");

   auto ls = new REX::REveStraightLineSet("model");
   ls->AddLine(0, 0, 0, 10, 10, 10);
   src->AddElement(ls);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   dst->AddElement(mng);
   mng->RefMainTrans().SetPos(0, 0, 50);

   auto other = new REX::REveElement("other");
   dst->AddElement(other);

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      ASSERT_NE(mng->ImportElements(ls), nullptr);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "REveProjectionManager::ImportElements", "'other' places projected elements", false);
      ASSERT_NE(mng->ImportElements(ls, other), nullptr);
   }
}

// The projection takes a model's coordinates through the matrices of the model
// and of its ancestors, as the 3D view does, for an import and for a child
// projected later through ProjectChild(). RPhi without distortion keeps x and
// y, so the projected points are the world points.
TEST(REveProjection, ModelTransIncludesAncestors)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("mt_src");
   auto dst = eve->SpawnNewScene("mt_dst");

   auto holder = new REX::REveCompound("holder");
   holder->RefMainTrans().SetPos(100, 0, 0);
   src->AddElement(holder);

   auto ps = new REX::REvePointSet("model");
   ps->SetNextPoint(1, 2, 0);
   ps->RefMainTrans().SetPos(0, 10, 0);
   holder->AddElement(ps);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   ASSERT_NE(mng->ImportElements(holder, dst), nullptr);
   ASSERT_EQ(ps->RefProjecteds().size(), 1u);
   auto pr = dynamic_cast<REX::REvePointSet *>(ps->RefProjecteds().front());
   ASSERT_NE(pr, nullptr);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fX, 101.f);
   EXPECT_FLOAT_EQ(pr->RefPoint(0).fY, 12.f);

   auto ps2 = new REX::REvePointSet("later");
   ps2->SetNextPoint(3, 4, 0);
   holder->AddElement(ps2);
   holder->ProjectChild(ps2);
   ASSERT_EQ(ps2->RefProjecteds().size(), 1u);
   auto pr2 = dynamic_cast<REX::REvePointSet *>(ps2->RefProjecteds().front());
   ASSERT_NE(pr2, nullptr);
   EXPECT_FLOAT_EQ(pr2->RefPoint(0).fX, 103.f);
   EXPECT_FLOAT_EQ(pr2->RefPoint(0).fY, 4.f);
}

// The projection is computed in the frame of the manager: a matrix on the
// manager's chain is inverted. Without one, a model whose only matrix is its
// own gets that matrix itself, as from PtrMainTrans(kFALSE).
TEST(REveProjection, ModelTransInManagerFrame)
{
   REX::REveManager::Create();
   auto holder = new REX::REveElement("holder");
   holder->RefMainTrans().SetPos(100, 0, 0);
   auto ps = new REX::REvePointSet("model");
   ps->RefMainTrans().SetPos(0, 10, 0);
   holder->AddElement(ps);
   auto lone = new REX::REvePointSet("lone");
   lone->RefMainTrans().SetPos(5, 0, 0);
   auto bare = new REX::REvePointSet("bare");

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   REX::REveTrans t;
   EXPECT_EQ(mng->GetModelTrans(bare, t), nullptr);
   EXPECT_EQ(mng->GetModelTrans(lone, t), lone->PtrMainTrans(kFALSE));

   auto m = mng->GetModelTrans(ps, t);
   ASSERT_EQ(m, &t);
   EXPECT_DOUBLE_EQ(m->GetPos().X(), 100.);
   EXPECT_DOUBLE_EQ(m->GetPos().Y(), 10.);

   mng->RefMainTrans().SetPos(100, 0, 0);
   m = mng->GetModelTrans(ps, t);
   ASSERT_EQ(m, &t);
   EXPECT_NEAR(m->GetPos().X(), 0., 1e-12);
   EXPECT_DOUBLE_EQ(m->GetPos().Y(), 10.);
}

// A projected jet cone takes the matrices of the model and its ancestors, in
// its vertices and in its bounding box. RPhi without distortion is linear, so a
// jet under a holder moved by 100 in x projects to the plain jet moved by 100.
TEST(REveProjection, JetConeModelTrans)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("jet_src");
   auto dst = eve->SpawnNewScene("jet_dst");

   auto make_jet = [](const char *name) {
      auto jet = new REX::REveJetCone(name);
      jet->SetCylinder(250, 300);
      jet->AddEllipticCone(0.5, 1.0, 0.1, 0.2);
      return jet;
   };
   auto plain = make_jet("plain");
   src->AddElement(plain);
   auto holder = new REX::REveElement("holder");
   holder->RefMainTrans().SetPos(100, 0, 0);
   src->AddElement(holder);
   auto moved = make_jet("moved");
   holder->AddElement(moved);

   auto mng = new REX::REveProjectionManager(REX::REveProjection::kPT_RPhi);
   mng->ImportElements(plain, dst);
   mng->ImportElements(holder, dst);
   auto pp = dynamic_cast<REX::REveElement *>(plain->RefProjecteds().front());
   auto pm = dynamic_cast<REX::REveElement *>(moved->RefProjecteds().front());
   ASSERT_NE(pp, nullptr);
   ASSERT_NE(pm, nullptr);

   auto vertices = [](REX::REveElement *el) {
      el->BuildRenderData();
      auto rd = el->GetRenderData();
      std::vector<float> v(rd->SizeV());
      std::vector<char> buf(rd->GetBinarySize());
      rd->Write(buf.data(), buf.size());
      std::copy_n(reinterpret_cast<float *>(buf.data()), v.size(), v.begin());
      return v;
   };
   auto vp = vertices(pp), vm = vertices(pm);
   ASSERT_EQ(vp.size(), vm.size());
   ASSERT_GE(vp.size(), 9u);
   for (size_t i = 0; i < vp.size(); i += 3) {
      EXPECT_NEAR(vm[i], vp[i] + 100, 1e-3) << "vertex " << i / 3;
      EXPECT_NEAR(vm[i + 1], vp[i + 1], 1e-3) << "vertex " << i / 3;
   }

   ASSERT_EQ(pp->AssertBBox(), REX::REveElement::kBBoxSet);
   ASSERT_EQ(pm->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_NEAR(pm->GetBBoxMin().fX, pp->GetBBoxMin().fX + 100, 1e-3);
   EXPECT_NEAR(pm->GetBBoxMax().fX, pp->GetBBoxMax().fX + 100, 1e-3);
   EXPECT_NEAR(pm->GetBBoxMin().fY, pp->GetBBoxMin().fY, 1e-3);
   EXPECT_NEAR(pm->GetBBoxMax().fY, pp->GetBBoxMax().fY, 1e-3);
}

// A projected geo shape takes the matrices of the shape and of its ancestors,
// for a 2D projection (REvePolygonSetProjected) and for a 3D one
// (REveGeoShapeProjected). Neither RPhi without distortion nor the 3D
// projection moves x and y, so the projected box is the box of the world
// points: 5 x 6 half-sizes around (100, 10).
TEST(REveProjection, GeoShapeModelTrans)
{
   auto eve = REX::REveManager::Create();
   auto src = eve->SpawnNewScene("geo_src");

   auto holder = new REX::REveElement("holder");
   holder->RefMainTrans().SetPos(100, 0, 0);
   src->AddElement(holder);
   auto shape = new REX::REveGeoShape("box");
   {
      // Register the shape with REve's geo manager, see GeoShapeBBoxAboutOrigin.
      REX::REveGeoManagerHolder gmgr(REX::REveGeoShape::GetGeoManager());
      shape->SetShape(new TGeoBBox(5, 6, 7));
   }
   shape->RefMainTrans().SetPos(0, 10, 0);
   holder->AddElement(shape);

   for (auto type : {REX::REveProjection::kPT_RPhi, REX::REveProjection::kPT_3D}) {
      auto dst = eve->SpawnNewScene("geo_dst");
      auto mng = new REX::REveProjectionManager(type);
      mng->ImportElements(holder, dst);
      auto pr = dynamic_cast<REX::REveElement *>(shape->RefProjecteds().back());
      ASSERT_NE(pr, nullptr);
      ASSERT_EQ(pr->AssertBBox(), REX::REveElement::kBBoxSet) << "projection " << type;
      EXPECT_NEAR(pr->GetBBoxMin().fX, 95.f, 1e-3) << "projection " << type;
      EXPECT_NEAR(pr->GetBBoxMax().fX, 105.f, 1e-3) << "projection " << type;
      EXPECT_NEAR(pr->GetBBoxMin().fY, 4.f, 1e-3) << "projection " << type;
      EXPECT_NEAR(pr->GetBBoxMax().fY, 16.f, 1e-3) << "projection " << type;
   }
}
