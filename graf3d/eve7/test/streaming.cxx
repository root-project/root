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
#include <ROOT/REveStraightLineSet.hxx>
#include <ROOT/REveText.hxx>
#include <ROOT/REveBox.hxx>
#include <ROOT/REveBoxSet.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveUtil.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveRenderData.hxx>

#include "TGeoTube.h"

#include "TBase64.h"

#include <nlohmann/json.hpp>

#include <cstring>

namespace REX = ROOT::Experimental;

namespace {

REX::REveScene *NewScene(const char *name)
{
   return REX::REveManager::Create()->SpawnNewScene(name);
}

REX::REvePointSet *MakePoints(const char *name, int n)
{
   auto ps = new REX::REvePointSet(name);
   for (int i = 0; i < n; ++i)
      ps->SetNextPoint(i, 2 * i, 3 * i);
   return ps;
}

} // namespace

// Core fields and render data of a point set; the layout depends on the
// renderer, RCore packs the points into a texture.
TEST(REveStreaming, PointSet)
{
   auto eve = REX::REveManager::Create();
   auto ps = MakePoints("points", 10);

   nlohmann::json j;
   int bin = ps->WriteCoreJson(j, 0);

   EXPECT_EQ(j["_typename"], "ROOT::Experimental::REvePointSet");
   EXPECT_EQ(j["fName"], "points");
   ASSERT_TRUE(j.contains("render_data"));
   auto &rd = j["render_data"];
   EXPECT_EQ(rd["rnr_func"], "makeHit");
   EXPECT_EQ(rd["rnr_offset"], 0);

   ASSERT_NE(ps->GetRenderData(), nullptr);
   EXPECT_EQ(bin, ps->GetRenderData()->GetBinarySize());
   EXPECT_EQ(bin % 4, 0);

   if (eve->IsRCore()) {
      // One texel of 4 floats per point.
      EXPECT_EQ(j["fSize"], 10);
      int tx = j["fTexX"], ty = j["fTexY"];
      EXPECT_GE(tx * ty, 10);
      EXPECT_EQ(rd["vert_size"], 4 * tx * ty);
      EXPECT_EQ(rd["norm_size"], 0);
   } else {
      EXPECT_EQ(rd["vert_size"], 30);
   }

   // The bounding box, min then max, in the core fields.
   ASSERT_TRUE(j.contains("bbox"));
   EXPECT_EQ(j["bbox"], nlohmann::json({0, 0, 0, 9, 18, 27}));

   delete ps;
}

// rnr_offset -1 asks for the core fields only, as for a colour or selection change.
TEST(REveStreaming, CoreFieldsOnly)
{
   REX::REveManager::Create();
   auto ps = MakePoints("points", 3);

   nlohmann::json j;
   EXPECT_EQ(ps->WriteCoreJson(j, -1), 0);
   EXPECT_FALSE(j.contains("render_data"));
   EXPECT_EQ(ps->GetRenderData(), nullptr);
   EXPECT_TRUE(j.contains("fMainColor"));

   delete ps;
}

TEST(REveStreaming, TextFields)
{
   REX::REveManager::Create();
   auto t = new REX::REveText("label");
   t->SetText("hello");
   t->SetFontSize(0.03f);
   t->SetTextAlign(REX::REveText::kCenterH, REX::REveText::kBottom);

   nlohmann::json j;
   EXPECT_EQ(t->WriteCoreJson(j, -1), 0);
   EXPECT_EQ(j["fText"], "hello");
   EXPECT_EQ(j["fFont"], "LiberationSerif-Regular");
   EXPECT_FLOAT_EQ(j["fFontSize"].get<float>(), 0.03f);
   EXPECT_EQ(j["fAlignH"], (int)REX::REveText::kCenterH);
   EXPECT_EQ(j["fAlignV"], (int)REX::REveText::kBottom);
   EXPECT_EQ(j["fClickMir"], "");

   delete t;
}

// A full scene stream: header, the scene, then its children in order, each
// with rnr_offset pointing at its block in the one binary message.
TEST(REveStreaming, SceneStream)
{
   auto s = NewScene("stream_test");

   auto ps = MakePoints("points", 5);
   auto ls = new REX::REveStraightLineSet("lines");
   ls->AddLine(0, 0, 0, 1, 1, 1);
   ls->AddLine(0, 0, 0, -1, 1, -1);
   s->AddElement(ps);
   s->AddElement(ls);
   EXPECT_NE(ps->GetElementId(), 0u);
   EXPECT_NE(ls->GetElementId(), 0u);

   s->StreamElements();

   auto j = nlohmann::json::parse(s->GetOutputJson());
   ASSERT_TRUE(j.is_array());
   // Header, scene, two elements, and the trailer with the server time.
   ASSERT_EQ(j.size(), 5u);
   EXPECT_TRUE(j[4]["server_time"].is_number());

   EXPECT_EQ(j[0]["content"], "REveScene::StreamElements");
   EXPECT_EQ(j[0]["fSceneId"], s->GetElementId());
   EXPECT_EQ(j[1]["fElementId"], s->GetElementId());
   EXPECT_EQ(j[2]["fElementId"], ps->GetElementId());
   EXPECT_EQ(j[3]["fElementId"], ls->GetElementId());
   for (int i : {2, 3}) {
      EXPECT_EQ(j[i]["fMotherId"], s->GetElementId());
      EXPECT_EQ(j[i]["fSceneId"], s->GetElementId());
   }

   int ps_bin = ps->GetRenderData()->GetBinarySize();
   int ls_bin = ls->GetRenderData()->GetBinarySize();
   EXPECT_EQ(j[0]["fTotalBinarySize"], ps_bin + ls_bin);
   EXPECT_EQ((int)s->GetOutputBinary().size(), ps_bin + ls_bin);
   EXPECT_EQ(j[2]["render_data"]["rnr_offset"], 0);
   EXPECT_EQ(j[3]["render_data"]["rnr_offset"], ps_bin);

   // The binary starts with the vertices of the first element.
   float v[3];
   std::memcpy(v, s->GetOutputBinary().data(), sizeof(v));
   EXPECT_FLOAT_EQ(v[0], 0.f);
   std::memcpy(v, s->GetOutputBinary().data() + 4 * sizeof(float), sizeof(v));
   if (REX::gEve->IsRCore()) {
      // Second texel: point 1.
      EXPECT_FLOAT_EQ(v[0], 1.f);
      EXPECT_FLOAT_EQ(v[1], 2.f);
      EXPECT_FLOAT_EQ(v[2], 3.f);
   }
}

// Stamping kCBObjProps invalidates the bounding box; the next AssertBBox()
// computes it from the current geometry.
TEST(REveStreaming, BBoxInvalidatedByObjProps)
{
   REX::REveManager::Create();
   auto ps = MakePoints("points", 3);

   EXPECT_EQ(ps->GetBBoxState(), REX::REveElement::kBBoxInvalid);
   ASSERT_EQ(ps->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(ps->GetBBoxMax().fZ, 6.f);

   ps->SetNextPoint(10, 20, 30);
   ps->StampObjProps();
   EXPECT_EQ(ps->GetBBoxState(), REX::REveElement::kBBoxInvalid);
   ASSERT_EQ(ps->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(ps->GetBBoxMax().fZ, 30.f);

   delete ps;
}

// The box covers the points set, not the capacity: SetNextPoint() doubles it
// with zero-filled slots, and Reset() keeps the old values.
TEST(REveStreaming, PointSetBBoxCoversPointsOnly)
{
   REX::REveManager::Create();
   auto ps = new REX::REvePointSet("points");
   for (int i = 0; i < 3; ++i)
      ps->SetNextPoint(10 + i, 20 + i, 30 + i);

   ASSERT_EQ(ps->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(ps->GetBBoxMin().fX, 10.f);
   EXPECT_FLOAT_EQ(ps->GetBBoxMin().fZ, 30.f);

   ps->Reset(8);
   ps->SetNextPoint(5, 5, 5);
   ASSERT_EQ(ps->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(ps->GetBBoxMax().fX, 5.f);
   EXPECT_FLOAT_EQ(ps->GetBBoxMin().fX, 5.f);

   delete ps;
}

// An empty element has no box, so the projection manager's box does not take
// it in, and it sends neither render data nor a box.
TEST(REveStreaming, EmptyElementsHaveNoBox)
{
   REX::REveManager::Create();
   auto ps = new REX::REvePointSet("points");
   auto ls = new REX::REveStraightLineSet("lines");

   for (REX::REveElement *e : {(REX::REveElement *)ps, (REX::REveElement *)ls}) {
      EXPECT_EQ(e->AssertBBox(), REX::REveElement::kBBoxNone);
      nlohmann::json j;
      EXPECT_EQ(e->WriteCoreJson(j, 0), 0);
      EXPECT_FALSE(j.contains("render_data"));
      EXPECT_FALSE(j.contains("bbox"));
   }

   delete ps;
   delete ls;
}

// Text has no box, and the client is told so.
TEST(REveStreaming, TextNotInWorld)
{
   REX::REveManager::Create();
   auto t = new REX::REveText("label");
   t->SetText("hello");

   nlohmann::json j;
   t->WriteCoreJson(j, 0);
   EXPECT_FALSE(j.contains("bbox"));
   EXPECT_EQ(j["bbox_not_in_world"], true);

   delete t;
}

// An REveBox streams its transformation as JSON, base64 of 16 doubles, and its
// box stays in its own frame.
TEST(REveStreaming, BoxTransAndBBox)
{
   REX::REveManager::Create();
   auto b = new REX::REveBox("box");
   const float h = 0.5f;
   const float v[8][3] = {{-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {h, -h, -h},
                          {-h, -h, h},  {-h, h, h},  {h, h, h},  {h, -h, h}};
   for (int i = 0; i < 8; ++i)
      b->SetVertex(i, v[i][0], v[i][1], v[i][2]);
   b->RefMainTrans().SetPos(100, 200, 300);

   nlohmann::json j;
   b->WriteCoreJson(j, 0);
   ASSERT_TRUE(j.contains("render_data"));
   EXPECT_FALSE(j["render_data"].contains("trans_size"));
   ASSERT_TRUE(j["matrix"].is_string());
   const TString bytes = TBase64::Decode(j["matrix"].get<std::string>().c_str());
   ASSERT_EQ(bytes.Length(), 16 * (int)sizeof(Double_t));
   Double_t m[16];
   std::memcpy(m, bytes.Data(), sizeof(m));
   EXPECT_DOUBLE_EQ(m[12], 100);
   EXPECT_DOUBLE_EQ(m[13], 200);
   EXPECT_DOUBLE_EQ(m[14], 300);
   EXPECT_EQ(j["bbox"], nlohmann::json({-h, -h, -h, h, h, h}));

   delete b;
}

// A geo shape's box is the TGeoBBox of its shape, about the shape's origin on
// each axis. A tube segment from 0 to 90 degrees has its origin off zero in x
// and y.
TEST(REveStreaming, GeoShapeBBoxAboutOrigin)
{
   REX::REveManager::Create();
   auto gs = new REX::REveGeoShape("tubeseg");
   {
      // Register the shape with REve's geo manager, which REveGeoShape deletes
      // it from; another manager would keep a dangling pointer to it.
      REX::REveGeoManagerHolder gmgr(REX::REveGeoShape::GetGeoManager());
      gs->SetShape(new TGeoTubeSeg(5, 10, 3, 0, 90));
   }

   ASSERT_EQ(gs->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(gs->GetBBoxMin().fX, 0.f);
   EXPECT_FLOAT_EQ(gs->GetBBoxMin().fY, 0.f);
   EXPECT_FLOAT_EQ(gs->GetBBoxMin().fZ, -3.f);
   EXPECT_FLOAT_EQ(gs->GetBBoxMax().fX, 10.f);
   EXPECT_FLOAT_EQ(gs->GetBBoxMax().fY, 10.f);
   EXPECT_FLOAT_EQ(gs->GetBBoxMax().fZ, 3.f);

   delete gs;
}

// The box of an instanced box set holds what the client draws: each instance
// at its position, reaching ShapeSize times its scale, or its matrix applied to
// ShapeSize, either way.
TEST(REveStreaming, BoxSetBBoxHoldsInstances)
{
   REX::REveManager::Create();
   auto expectBox = [](REX::REveElement *e, REX::REveVector lo, REX::REveVector hi) {
      ASSERT_EQ(e->AssertBBox(), REX::REveElement::kBBoxSet);
      for (int k = 0; k < 3; ++k) {
         EXPECT_NEAR(e->GetBBoxMin()[k], lo[k], 1e-4) << e->GetCName() << " min " << k;
         EXPECT_NEAR(e->GetBBoxMax()[k], hi[k], 1e-4) << e->GetCName() << " max " << k;
      }
   };

   auto fixed = new REX::REveBoxSet("fixed");
   fixed->Reset(REX::REveBoxSet::kBT_Instanced, kFALSE, 4);
   fixed->SetDefHalfWidth(2);
   fixed->SetDefHeight(2);
   fixed->SetDefDepth(2);
   fixed->AddInstance(10, 10, 10);
   expectBox(fixed, {8, 9, 9}, {12, 11, 11});

   auto scaled = new REX::REveBoxSet("scaled");
   scaled->Reset(REX::REveBoxSet::kBT_InstancedScaled, kFALSE, 4);
   // Each instance adds its own box: the small one at -10 does not take the
   // half sizes of the large one.
   scaled->AddInstanceScaled(10, 10, 10, 1, 2, 3);
   scaled->AddInstanceScaled(-10, 0, 0, 1, 1, 1);
   expectBox(scaled, {-11, -1, -1}, {11, 12, 13});

   // A cone with its apex at the origin, 60 long along y, radius 5. Its box is
   // that of the template cube [-1, 1]^3 under the cone's matrix.
   auto cone = new REX::REveBoxSet("cone");
   cone->Reset(REX::REveBoxSet::kBT_InstancedScaledRotated, kFALSE, 4);
   cone->AddCone(REX::REveVector(0, 0, 0), REX::REveVector(0, 60, 0), 5);
   expectBox(cone, {-5, -60, -5}, {5, 60, 5});

   delete fixed;
   delete scaled;
   delete cone;
}

// An axis-aligned instance added by its corner and full size is stored as its
// centre and half size, which the client draws. The set's box is the union of
// the instances' boxes.
TEST(REveStreaming, BoxSetCornerAdders)
{
   REX::REveManager::Create();
   auto expectBox = [](REX::REveElement *e, REX::REveVector lo, REX::REveVector hi) {
      ASSERT_EQ(e->AssertBBox(), REX::REveElement::kBBoxSet);
      for (int k = 0; k < 3; ++k) {
         EXPECT_NEAR(e->GetBBoxMin()[k], lo[k], 1e-4) << e->GetCName() << " min " << k;
         EXPECT_NEAR(e->GetBBoxMax()[k], hi[k], 1e-4) << e->GetCName() << " max " << k;
      }
   };

   // SetDefWidth() & co. take the full size, SetDefHalfWidth() & co. the half.
   auto fixed = new REX::REveBoxSet("fixed");
   fixed->Reset(REX::REveBoxSet::kBT_Instanced, kFALSE, 4);
   EXPECT_FLOAT_EQ(fixed->GetDefWidth(), 1.f);
   EXPECT_FLOAT_EQ(fixed->GetDefHalfWidth(), 0.5f);
   fixed->SetDefWidth(4);
   fixed->SetDefHalfHeight(1);
   fixed->SetDefDepth(2);
   EXPECT_FLOAT_EQ(fixed->GetDefHalfWidth(), 2.f);
   EXPECT_FLOAT_EQ(fixed->GetDefHeight(), 2.f);
   fixed->AddInstanceCorner(10, 10, 10);
   expectBox(fixed, {10, 10, 10}, {14, 12, 12});

   // Boxes from (10, 10, 10) to (11, 12, 13) and from (-10, 0, 0) to (-9, 1, 1).
   auto scaled = new REX::REveBoxSet("scaled");
   scaled->Reset(REX::REveBoxSet::kBT_InstancedScaled, kFALSE, 4);
   scaled->UseSingleColor();
   scaled->AddInstanceScaledCorner(10, 10, 10, 1, 2, 3);
   scaled->AddInstanceScaledCorner(-10, 0, 0, 1, 1, 1);
   expectBox(scaled, {-10, 0, 0}, {11, 12, 13});

   // The first texel holds the centre and the second the half size.
   nlohmann::json j;
   const int bin = scaled->WriteCoreJson(j, 0);
   std::vector<float> buf(bin / sizeof(float));
   scaled->GetRenderData()->Write(reinterpret_cast<char *>(buf.data()), bin);
   EXPECT_FLOAT_EQ(buf[0], 10.5f);
   EXPECT_FLOAT_EQ(buf[1], 11.f);
   EXPECT_FLOAT_EQ(buf[2], 11.5f);
   EXPECT_FLOAT_EQ(buf[4], 0.5f);
   EXPECT_FLOAT_EQ(buf[6], 1.5f);

   delete fixed;
   delete scaled;
}

// An element emptied of its contents sends no render data, not the render data
// of what it held before.
TEST(REveStreaming, EmptiedElementSendsNoRenderData)
{
   REX::REveManager::Create();
   auto ps = MakePoints("points", 5);

   nlohmann::json j;
   ps->WriteCoreJson(j, 0);
   ASSERT_TRUE(j.contains("render_data"));

   ps->Reset(0);
   nlohmann::json je;
   EXPECT_EQ(ps->WriteCoreJson(je, 0), 0);
   EXPECT_FALSE(je.contains("render_data"));
   EXPECT_FALSE(je.contains("bbox"));
   EXPECT_EQ(ps->GetRenderData(), nullptr);

   delete ps;
}

// An element with no geometry of its own has no render data and sends no box.
namespace {
// Fills its box from no points at all.
class EmptyBoxElement : public REX::REveElement {
public:
   EmptyBoxElement() : REX::REveElement("empty") {}
   void ComputeBBox() override { BBoxInit(); }
};
} // namespace

// BBoxInit() starts from +-infinity: a box no point reached is reported as
// kBBoxNone and not streamed, and a point beyond any finite start sets the box.
TEST(REveStreaming, BBoxFromInfinity)
{
   REX::REveManager::Create();

   auto e = new EmptyBoxElement;
   EXPECT_EQ(e->AssertBBox(), REX::REveElement::kBBoxNone);

   auto ps = new REX::REvePointSet("far");
   ps->SetNextPoint(2e6, -3e6, 4e6);
   ASSERT_EQ(ps->AssertBBox(), REX::REveElement::kBBoxSet);
   EXPECT_FLOAT_EQ(ps->GetBBoxMin().fX, 2e6);
   EXPECT_FLOAT_EQ(ps->GetBBoxMax().fY, -3e6);
   EXPECT_FLOAT_EQ(ps->GetBBoxMin().fZ, 4e6);

   delete e;
   delete ps;
}

TEST(REveStreaming, GroupingElementNoBBox)
{
   REX::REveManager::Create();
   auto g = new REX::REveElement("group");

   nlohmann::json j;
   g->WriteCoreJson(j, 0);
   EXPECT_FALSE(j.contains("render_data"));
   EXPECT_FALSE(j.contains("bbox"));
   EXPECT_EQ(g->AssertBBox(), REX::REveElement::kBBoxNone);

   delete g;
}
