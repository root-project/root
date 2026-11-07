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

#include "TBase64.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
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

// A name change stamps kCBCoreJson, which streams the new name without
// render data.
TEST(REveChangeCycle, NameStreamsCoreJson)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_name");
   auto ps = MakePoints("points", 3);
   s->AddElement(ps);

   eve->BeginChange();
   ps->SetName("renamed");
   EXPECT_EQ(ps->GetChangeBits(), REX::REveElement::kCBCoreJson);
   auto out = EndChangeKeepingOutput(s);

   auto &j = out.fJson;
   ASSERT_EQ(j["arr"].size(), 1u);
   auto &e = j["arr"][0];
   EXPECT_EQ(e["fElementId"], ps->GetElementId());
   EXPECT_EQ(e["changeBit"], REX::REveElement::kCBCoreJson);
   EXPECT_EQ(e["fName"], "renamed");
   EXPECT_FALSE(e.contains("render_data"));
   EXPECT_EQ(out.fBinarySize, 0u);
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
   moved->StampTrans();
   edited->StampObjProps();

   nlohmann::json arr = nlohmann::json::array();
   s->StreamMotionChanges(arr);

   ASSERT_EQ(arr.size(), 1u);
   EXPECT_EQ(arr[0]["fElementId"], moved->GetElementId());
   EXPECT_EQ(arr[0]["fSceneId"], s->GetElementId());
   // Base64 of 16 doubles, column-major, translation in elements 12 to 14.
   ASSERT_TRUE(arr[0]["matrix"].is_string());
   const TString bytes = TBase64::Decode(arr[0]["matrix"].get<std::string>().c_str());
   ASSERT_EQ(bytes.Length(), 16 * (int)sizeof(Double_t));
   Double_t m[16];
   std::memcpy(m, bytes.Data(), sizeof(m));
   EXPECT_DOUBLE_EQ(m[12], 1.);
   EXPECT_DOUBLE_EQ(m[13], 2.);
   EXPECT_DOUBLE_EQ(m[14], 3.);
   EXPECT_TRUE(arr[0]["motion"].is_null());

   EXPECT_EQ(moved->GetChangeBits(), 0);
   EXPECT_NE(edited->GetChangeBits(), 0);
   auto out = EndChangeKeepingOutput(s);

   auto &j = out.fJson;
   ASSERT_EQ(j["arr"].size(), 1u);
   EXPECT_EQ(j["arr"][0]["fElementId"], edited->GetElementId());
   EXPECT_EQ(j["header"]["numRepresentationChanged"], 1);
}

// A full update writes the whole transformation, the motion included, so a
// motion set in the same cycle is not lost; without a motion "motion" is absent.
// The header carries the server time, the client's clock for motion.
TEST(REveChangeCycle, FullUpdateCarriesMotion)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_full_motion");
   auto ps = MakePoints("points", 2);
   s->AddElement(ps);

   eve->BeginChange();
   ps->RefMainTrans().SetMotion({0, 0, 10}, {0, 0, 0}, 1.);
   ps->StampTrans();
   ps->StampObjProps();
   auto out = EndChangeKeepingOutput(s);

   auto &j = out.fJson;
   EXPECT_TRUE(j["server_time"].is_number());
   ASSERT_EQ(j["arr"].size(), 1u);
   auto &e = j["arr"][0];
   EXPECT_TRUE(e["matrix"].is_string());
   ASSERT_TRUE(e["motion"].is_object());
   EXPECT_EQ(e["motion"]["vel"], nlohmann::json({0., 0., 10.}));

   eve->BeginChange();
   ps->RefMainTrans().ClearMotion();
   ps->StampObjProps();
   auto out2 = EndChangeKeepingOutput(s);
   ASSERT_EQ(out2.fJson["arr"].size(), 1u);
   EXPECT_FALSE(out2.fJson["arr"][0].contains("motion"));

   // The full stream for a new client carries the motion and the time too.
   ps->RefMainTrans().SetMotion({1, 0, 0}, {0, 0, 0}, 1.);
   s->StreamElements();
   auto full = nlohmann::json::parse(s->GetOutputJson());
   EXPECT_TRUE(full.back()["server_time"].is_number());
   bool found = false;
   for (auto &el : full)
      if (el.contains("fElementId") && el["fElementId"] == ps->GetElementId()) {
         found = true;
         EXPECT_TRUE(el["motion"].is_object());
      }
   EXPECT_TRUE(found);
}

// Creating and destroying the main transformation are transformation changes.
TEST(REveChangeCycle, MainTransInitAndDestroyStamp)
{
   auto eve = REX::REveManager::Create();
   auto s = NewScene("cc_trans_stamp");
   auto el = new REX::REveElement("holder");
   s->AddElement(el);

   eve->BeginChange();
   el->InitMainTrans();
   EXPECT_TRUE(el->GetChangeBits() & REX::REveElement::kCBTrans);
   eve->EndChange();

   eve->BeginChange();
   el->DestroyMainTrans();
   EXPECT_TRUE(el->GetChangeBits() & REX::REveElement::kCBTrans);
   eve->EndChange();
}
