// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveScene.hxx>
#include <ROOT/REveViewer.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveClient.hxx>
#include <ROOT/RWebWindow.hxx>

#include <cassert>

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveScene
\ingroup REve
Eve representation of TGLScene.
The GLScene is owned by this class - it is created on construction
time and deleted at destruction.

Normally all objects are positioned directly in global scene-space.
By setting the fHierarchical flag, positions of children get
calculated by multiplying the transformation matrices of all parents.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveScene::REveScene(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   fScene = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveScene::~REveScene()
{
   fDestructing = kStandard;

   REX::gEve->GetViewers()->SceneDestructing(this);
   REX::gEve->GetScenes()->RemoveElement(this);
}
//------------------------------------------------------------------------------

int REveScene::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   j["Mandatory"] = fMandatory;
   j["IsOverlay"] = fIsOverlay;

   return REveElement::WriteCoreJson(j, rnr_offset);
}

//------------------------------------------------------------------------------

void REveScene::AddSubscriber(std::unique_ptr<REveClient> &&sub)
{
   assert(sub.get() != nullptr && fAcceptingChanges == kFALSE);

   fSubscribers.emplace_back(std::move(sub));

   // XXX Here should send out the package to the new subscriber,
   // In principle can expect a new one in short time?
   // Keep streamed data until next begin change, maybe.
}

void REveScene::RemoveSubscriber(unsigned id)
{
   auto pred = [&](std::unique_ptr<REveClient> &client) {
      return client->fId == id;
   };

   fSubscribers.erase(std::remove_if(fSubscribers.begin(), fSubscribers.end(), pred), fSubscribers.end());
}

// Add Button in client gui with this command
void REveScene::AddCommand(const std::string &name, const std::string &icon, const REveElement *element, const std::string &action)
{
   static const REveException eh("REveScene::AddCommand ");
   if (element->GetElementId() && element->IsA())
   {
      fCommands.emplace_back(name, icon, element, action);
   }
   else
   {
      throw eh + "Element id and dictionary has to be defined";
   }
}

void REveScene::BeginAcceptingChanges()
{
   if (fAcceptingChanges) return;

   if (HasSubscribers()) {
      fAcceptingChanges = kTRUE;
      /*
      for (auto &&client : fSubscribers) {
         REX::gEve->SceneSubscriberProcessingChanges(client->fId);
      }
      */
   }
}

void REveScene::SceneElementChanged(REveElement* element)
{
   assert(fAcceptingChanges);

   fChangedElements.push_back(element);
}

void REveScene::SceneElementRemoved(ElementId_t id)
{
   fRemovedElements.push_back(id);
}

void REveScene::EndAcceptingChanges()
{
   if ( ! fAcceptingChanges) return;

   fAcceptingChanges = kFALSE;
}

void REveScene::StreamElements()
{
   fOutputJson.clear();
   fOutputBinary.clear();

   fElsWithBinaryData.clear();
   fTotalBinarySize = 0;

   nlohmann::json jarr = nlohmann::json::array();

   nlohmann::json jhdr = {};
   jhdr["content"]  = "REveScene::StreamElements";
   jhdr["fSceneId"] = fElementId;

   if (!fCommands.empty()) {
      jhdr["commands"] = nlohmann::json::array();
      for (auto &&cmd : fCommands) {
         nlohmann::json jcmd = {};
         jcmd["name"]  = cmd.fName;
         jcmd["icon"] = cmd.fIcon;
         jcmd["elementid"] = cmd.fElementId;
         jcmd["elementclass"] = cmd.fElementClass;
         jcmd["func"] = cmd.fAction; // SL: may be not needed on client side, can use name
         jhdr["commands"].push_back(jcmd);
      }
   }

   jarr.push_back(jhdr);

   StreamJsonRecurse(this, jarr);
   // for (auto &c : fChildren)
   // {
   //    StreamJsonRecurse(c, jarr);
   // }

   fOutputBinary.resize(fTotalBinarySize);
   Int_t off = 0;

   for (auto &&e : fElsWithBinaryData)
   {
      auto rd_size = e->fRenderData->Write(&fOutputBinary[off], fOutputBinary.size() - off);
      off += rd_size;
   }
   assert(off == fTotalBinarySize);

   jarr.front()["fTotalBinarySize"] = fTotalBinarySize;

   fOutputJson = jarr.dump();
}

void REveScene::StreamJsonRecurse(REveElement *el, nlohmann::json &jarr)
{
   nlohmann::json jobj = {};
   Int_t rd_size = el->WriteCoreJson(jobj, fTotalBinarySize);
   jarr.push_back(jobj);

   // If this is another scene, do not stream additional details.
   // It should be requested / subscribed to independently.

   if (el->fScene == el && el != this)
   {
      return;
   }

   if (rd_size > 0)
   {
      assert (rd_size % 4 == 0);

      fTotalBinarySize += rd_size;
      fElsWithBinaryData.push_back(el);
   }

   for (auto &&c : el->fChildren)
   {
      // Stream only objects element el is a mother of.
      //
      // XXXX This is spooky side effect of multi-parenting.
      //
      // In particular screwed up for selection.
      // Selection now streams element ids and implied selected ids
      // and secondary-ids as part of core json.
      //
      // I wonder how this screws up REveProjectionManager (should
      // we hold a map of already streamed ids?).
      //
      // Do uncles and aunts and figure out a clean way for backrefs.

      if (c->GetMother() == el)
      {
         StreamJsonRecurse(c, jarr);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
//
/// Prepare data for sending element changes
//
////////////////////////////////////////////////////////////////////////////////

void REveScene::StreamRepresentationChanges()
{
   fElsWithBinaryData.clear();
   fTotalBinarySize = 0;

   nlohmann::json jarr = nlohmann::json::array();

   nlohmann::json jhdr = {};
   jhdr["content"]  = "ElementsRepresentaionChanges";
   jhdr["fSceneId"] = fElementId;

   jhdr["removedElements"] = nlohmann::json::array();
   for (auto &re : fRemovedElements)
      jhdr["removedElements"].push_back(re);

   jhdr["numRepresentationChanged"] = fChangedElements.size();

   // jarr.push_back(jhdr);

   for (auto &el: fChangedElements)
   {
      UChar_t bits = el->GetChangeBits();

      nlohmann::json jobj = {};
      jobj["fElementId"] = el->GetElementId();
      jobj["changeBit"]  = bits;

      if (bits & kCBElementAdded || bits & kCBObjProps)
      {
         if (gDebug > 0 && bits & kCBElementAdded)
         {
            Info("REveScene::StreamRepresentationChanges", "new element change %s %d\n",
                 el->GetCName(), bits);
         }

         Int_t rd_size = el->WriteCoreJson(jobj, fTotalBinarySize);
         if (rd_size) {
            assert (rd_size % 4 == 0);
            fTotalBinarySize += rd_size;
            fElsWithBinaryData.push_back(el);
         }
      }
      else
      {
        if (bits & kCBVisibility)
        {
          jobj["fRnrSelf"]     = el->GetRnrSelf();
          jobj["fRnrChildren"] = el->GetRnrChildren();
        }

        if (bits & kCBColorSelection)
        {
          el->WriteCoreJson(jobj, -1);
        }

        if (bits & kCBTransBBox)
        {
        }
      }

      jarr.push_back(jobj);

      el->ClearStamps();
   }

   fChangedElements.clear();
   fRemovedElements.clear();

   // render data for total change
   fOutputBinary.resize(fTotalBinarySize);
   Int_t off = 0;

   for (auto &e : fElsWithBinaryData) {
      auto rd_size = e->fRenderData->Write(&fOutputBinary[off], fOutputBinary.size() - off);

      off += rd_size;
   }
   assert(off == fTotalBinarySize);

   jhdr["fTotalBinarySize"] = fTotalBinarySize;

   nlohmann::json msg = { {"header", jhdr}, {"arr", jarr}};
   fOutputJson = msg.dump();

   if (gDebug > 0)
      Info("REveScene::StreamRepresentationChanges", "class: %s  changes %s ...", GetCName(),  msg.dump(1).c_str() );
}

void REveScene::SendChangesToSubscribers()
{
   for (auto && client : fSubscribers) {
      if (!fOutputJson.empty()) {
         if (gDebug > 0)
            printf("   sending json, len = %d --> to conn_id = %d\n", (int) fOutputJson.size(), client->fId);
         client->fWebWindow->Send(client->fId, fOutputJson);
      }
      if (fTotalBinarySize) {
         if (gDebug > 0)
            printf("   sending binary, len = %d --> to conn_id = %d\n", fTotalBinarySize, client->fId);
         client->fWebWindow->SendBinary(client->fId, &fOutputBinary[0], fTotalBinarySize);
      }
      REX::gEve->SceneSubscriberWaitingResponse(client->fId);
   }
   fOutputJson.clear();
   fOutputBinary.clear();
   fTotalBinarySize = 0;
}

Bool_t REveScene::IsChanged() const
{
   if (gDebug > 0)
     ::Info("REveScene::IsChanged","%s (changed_or_added=%d, removed=%d)", GetCName(),
          (int) fChangedElements.size(), (int) fRemovedElements.size());

   return ! (fChangedElements.empty() && fRemovedElements.empty());
}


/** \class REveSceneList
\ingroup REve
List of Scenes providing common operations on REveScene collections.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveSceneList::REveSceneList(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   SetChildClass(TClass::GetClass<REveScene>());
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy all scenes and their contents.
/// The object with non-zero deny-destroy will still survive.

void REveSceneList::DestroyScenes()
{
   auto i = fChildren.begin();
   while (i != fChildren.end())
   {
      REveScene* s = (REveScene*) *(i++);
      s->DestroyElements();
      s->DestroyOrWarn();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Loop-wrapers over Scene children, element type checked on insertion.

void REveSceneList::BeginAcceptingChanges()
{
   for (auto &c: fChildren)
   {
      static_cast<REveScene*>(c)->BeginAcceptingChanges();
   }
}

void REveSceneList::EndAcceptingChanges()
{
   for (auto &c: fChildren)
   {
      static_cast<REveScene*>(c)->EndAcceptingChanges();
   }
}

bool REveSceneList::AnyChanges() const
{
   for (auto &el : fChildren)
   {
      if (((REveScene*) el)->IsChanged())
      return true;
   }
   return false;
}
