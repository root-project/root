// @(#)root/eve7:$Id$

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// The change cycle without a browser: REveManager::BeginChange() and
// EndChange() with no connection, so EndChange() streams the scene changes
// and sends them to nobody.

#include "gtest/gtest.h"

#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveClient.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveTrans.hxx>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <memory>

namespace REX = ROOT::Experimental;

namespace {

/// A scene accepts changes only while it has a subscriber. This one is
/// subscribed under a connection id the web window does not have, so the
/// changes are streamed and the send reaches nobody.
REX::REveScene *NewScene(const char *name)
{
   auto eve = REX::REveManager::Create();
   auto s = eve->SpawnNewScene(name);
   auto win = eve->GetWebWindow();
   s->AddSubscriber(std::make_unique<REX::REveClient>(999999u, win));
   return s;
}

REX::REvePointSet *MakePoints(const char *name, int n)
{
   auto ps = new REX::REvePointSet(name);
   for (int i = 0; i < n; ++i)
      ps->SetNextPoint(i, i, i);
   return ps;
}

struct Streamed {
   nlohmann::json fJson;
   size_t fBinarySize{0};
};

/// Streams the scene's changes in the order EndChange() does, motion first,
/// keeps the result, then ends the change cycle. The result cannot be read
/// after EndChange(): sending the changes clears the scene's output.
Streamed EndChangeKeepingOutput(REX::REveScene *s)
{
   s->EndAcceptingChanges();
   nlohmann::json motion = nlohmann::json::array();
   s->StreamMotionChanges(motion);
   s->StreamRepresentationChanges();
   Streamed out{nlohmann::json::parse(s->GetOutputJson()), s->GetOutputBinary().size()};
   REX::REveManager::Create()->EndChange();
   return out;
}

bool Contains(const nlohmann::json &arr, REX::ElementId_t id)
{
   return std::find(arr.begin(), arr.end(), id) != arr.end();
}

} // namespace

TEST(REveChangeCycle, StampsOnlyDuringChange)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_stamps");
   auto ps = MakePoints("points", 3);
   s->AddElement(ps);

   ps->StampObjProps();
   EXPECT_EQ(ps->GetChangeBits(), 0);
   EXPECT_FALSE(s->IsChanged());

   eve->BeginChange();
   ps->StampObjProps();
   EXPECT_TRUE(ps->GetChangeBits() & REX::REveElement::kCBObjProps);
   EXPECT_TRUE(s->IsChanged());
   eve->EndChange();

   EXPECT_EQ(ps->GetChangeBits(), 0);
   EXPECT_FALSE(s->IsChanged());
}

TEST(REveChangeCycle, AddedElementIsStreamed)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_added");

   eve->BeginChange();
   auto ps = MakePoints("points", 4);
   s->AddElement(ps);
   EXPECT_TRUE(ps->GetChangeBits() & REX::REveElement::kCBElementAdded);
   auto out = EndChangeKeepingOutput(s);

   auto &j = out.fJson;
   auto &hdr = j["header"];
   EXPECT_EQ(hdr["content"], "ElementsRepresentaionChanges");
   EXPECT_EQ(hdr["fSceneId"], s->GetElementId());
   EXPECT_TRUE(hdr["removedElements"].empty());

   ASSERT_EQ(j["arr"].size(), 1u);
   auto &e = j["arr"][0];
   EXPECT_EQ(e["fElementId"], ps->GetElementId());
   EXPECT_TRUE(e["changeBit"].get<int>() & REX::REveElement::kCBElementAdded);
   ASSERT_TRUE(e.contains("render_data"));

   int bin = ps->GetRenderData()->GetBinarySize();
   EXPECT_EQ(hdr["fTotalBinarySize"], bin);
   EXPECT_EQ((int)out.fBinarySize, bin);
}

// The id reported as removed is the removed element's, not its mother's.
TEST(REveChangeCycle, RemovedElementIsReported)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_removed");
   auto holder = new REX::REveElement("holder");
   auto ps = MakePoints("points", 2);
   s->AddElement(holder);
   holder->AddElement(ps);
   const auto ps_id = ps->GetElementId();

   eve->BeginChange();
   holder->RemoveElement(ps);
   auto out = EndChangeKeepingOutput(s);

   auto &removed = out.fJson["header"]["removedElements"];
   EXPECT_TRUE(Contains(removed, ps_id));
   EXPECT_FALSE(Contains(removed, holder->GetElementId()));
}

// A transformation-only change leaves the change cycle and goes to the motion
// channel; other changes in the same cycle are streamed as usual.
TEST(REveChangeCycle, TransOnlyGoesToMotion)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_motion");
   auto moved = MakePoints("moved", 2);
   auto edited = MakePoints("edited", 2);
   s->AddElement(moved);
   s->AddElement(edited);

   eve->BeginChange();
   moved->RefMainTrans().SetPos(1, 2, 3);
   moved->StampTransBBox();
   edited->StampObjProps();

   nlohmann::json arr = nlohmann::json::array();
   s->StreamMotionChanges(arr);

   ASSERT_EQ(arr.size(), 1u);
   EXPECT_EQ(arr[0]["fElementId"], moved->GetElementId());
   EXPECT_EQ(arr[0]["fSceneId"], s->GetElementId());
   ASSERT_EQ(arr[0]["matrix"].size(), 16u);
   // Column-major, translation in elements 12 to 14.
   EXPECT_DOUBLE_EQ(arr[0]["matrix"][12].get<double>(), 1.);
   EXPECT_DOUBLE_EQ(arr[0]["matrix"][13].get<double>(), 2.);
   EXPECT_DOUBLE_EQ(arr[0]["matrix"][14].get<double>(), 3.);
   EXPECT_TRUE(arr[0]["mot"].is_null());

   EXPECT_EQ(moved->GetChangeBits(), 0);
   EXPECT_NE(edited->GetChangeBits(), 0);
   auto out = EndChangeKeepingOutput(s);

   auto &j = out.fJson;
   ASSERT_EQ(j["arr"].size(), 1u);
   EXPECT_EQ(j["arr"][0]["fElementId"], edited->GetElementId());
   EXPECT_EQ(j["header"]["numRepresentationChanged"], 1);
}
