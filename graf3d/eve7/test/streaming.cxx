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
#include <ROOT/REveRenderData.hxx>

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
      // One texel of 4 floats per point, and the bbox, min then max, as normals.
      EXPECT_EQ(j["fSize"], 10);
      int tx = j["fTexX"], ty = j["fTexY"];
      EXPECT_GE(tx * ty, 10);
      EXPECT_EQ(rd["vert_size"], 4 * tx * ty);
      EXPECT_EQ(rd["norm_size"], 6);
   } else {
      EXPECT_EQ(rd["vert_size"], 30);
   }

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
   ASSERT_EQ(j.size(), 4u);

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
