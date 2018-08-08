// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TEveScene.hxx"
#include "ROOT/TEveViewer.hxx"
#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveTrans.hxx"
#include <ROOT/TWebWindowsManager.hxx>

#include "TList.h"
#include "TExMap.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class TEveScene
\ingroup TEve
Eve representation of TGLScene.
The GLScene is owned by this class - it is created on construction
time and deleted at destruction.

Normally all objects are positioned directly in global scene-space.
By setting the fHierarchical flag, positions of children get
calculated by multiplying the transformation matrices of all parents.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveScene::TEveScene(const char* n, const char* t) :
   TEveElementList(n, t)
{
   fScene = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveScene::~TEveScene()
{
   fDestructing = kStandard;

   REX::gEve->GetViewers()->SceneDestructing(this);
   REX::gEve->GetScenes()->RemoveElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement; here we simply append this scene to
/// the list.

void TEveScene::CollectSceneParents(List_t& scenes)
{
   scenes.push_back(this);
}

//------------------------------------------------------------------------------

void TEveScene::AddSubscriber(TEveClient* sub)
{
   assert(sub != 0 && fAcceptingChanges == kFALSE);

   fSubscribers.push_back(sub);

   // XXX Here should send out the package to the new subscriber,
   // In principle can expect a new one in short time?
   // Keep streamed data until next begin change, maybe.
}

void TEveScene::RemoveSubscriber(unsigned id)
{
   assert(fAcceptingChanges == kFALSE);
   for (auto &client : fSubscribers) {
      if (client->fId == id ) {
         fSubscribers.remove(client);
         delete client;
      }
   }
}

void TEveScene::BeginAcceptingChanges()
{
   if (fAcceptingChanges) return;

   if (HasSubscribers()) fAcceptingChanges = kTRUE;
}

void TEveScene::SceneElementChanged(TEveElement* element)
{
   assert(fAcceptingChanges);

   fChangedElements.insert(element);
}

void TEveScene::EndAcceptingChanges()
{
   if ( ! fAcceptingChanges) return;

   fAcceptingChanges = kFALSE;
}

void TEveScene::ProcessChanges()
{
   // should return net message or talk to gEve about it
}

void TEveScene::StreamElements()
{
   fOutputJson.clear();
   fOutputBinary.clear();

   fElsWithBinaryData.clear();
   fTotalBinarySize = 0;

   nlohmann::json jarr = nlohmann::json::array();

   nlohmann::json jhdr = {};
   jhdr["content"]  = "TEveScene::StreamElements";
   jhdr["fSceneId"] = fElementId;
   jarr.push_back(jhdr);                       \

   StreamJsonRecurse(this, jarr);
   // for (auto &c : fChildren)
   // {
   //    StreamJsonRecurse(c, jarr);
   // }

   fOutputBinary.resize(fTotalBinarySize);
   Int_t actual_binary_size = 0;

   for (auto &e : fElsWithBinaryData)
   {
      Int_t rd_size = e->fRenderData->Write( & fOutputBinary[ actual_binary_size ] );

      actual_binary_size += rd_size;
   }
   assert(actual_binary_size == fTotalBinarySize);

   jarr.front()["fTotalBinarySize"] = fTotalBinarySize;

   fOutputJson = jarr.dump();
}

void TEveScene::StreamJsonRecurse(TEveElement *el, nlohmann::json &jarr)
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

   for (auto &c : el->fChildren)
   {
      StreamJsonRecurse(c, jarr);
   }
}

////////////////////////////////////////////////////////////////////////////////
//
/// Prepare data for sending element changes
//
////////////////////////////////////////////////////////////////////////////////
void TEveScene::StreamRepresentationChanges()
{     
   fOutputJson.clear();
   fOutputBinary.clear();
   fElsWithBinaryData.clear();
   fTotalBinarySize = 0;
   
   nlohmann::json jarr = nlohmann::json::array();

   nlohmann::json jhdr = {};
   jhdr["content"]  = "ElementsRepresentaionChanges";
   jhdr["fSceneId"] = fElementId;
   jarr.push_back(jhdr);

   for (Set_i i = fChangedElements.begin(); i != fChangedElements.end(); ++i)
   {
      TEveElement* el = *i;
      UChar_t bits = el->GetChangeBits();

      nlohmann::json jobj = {};
      jobj["fElementId"] = el->GetElementId();
      if (bits & kCBVisibility)
      {
         jobj["fRnrSelf"]     = el->GetRnrSelf();
         jobj["fRnrchildren"] = el->GetRnrChildren();
      }

      if (bits & kCBColorSelection)
      {
         jobj["fMainColor"] = el->GetMainColor();
      }

      if (bits & kCBTransBBox)
      {
      }

      if (bits & kCBObjProps)
      {
         printf("total element chamge %s \n", el->GetElementName());
         el->WriteCoreJson(jobj, fTotalBinarySize);
         fElsWithBinaryData.push_back(el);
      }

      jarr.push_back(jobj);

      el->ClearStamps();
   }
   fChangedElements.clear();
   fOutputJson = jarr.dump();


   // render data for total change
   fOutputBinary.resize(fTotalBinarySize);
   Int_t actual_binary_size = 0;

   for (auto &e : fElsWithBinaryData)
   {
      Int_t rd_size = e->fRenderData->Write( & fOutputBinary[ actual_binary_size ] );

      actual_binary_size += rd_size;
   }
   assert(actual_binary_size == fTotalBinarySize);

   jarr.front()["fTotalBinarySize"] = fTotalBinarySize;


   printf("[%s] Stream representation changes %s \n", GetElementName(), fOutputJson.c_str() );
}

void
TEveScene::SendChangesToSubscribers()
{
   for (auto & client : fSubscribers) {
      printf("   sending json, len = %d --> to conn_id = %d\n", (int) fOutputJson.size(), client->fId);
      client->fWebWindow->Send(client->fId, fOutputJson);
      if (fTotalBinarySize) {
         printf("   sending binary, len = %d --> to conn_id = %d\n",fTotalBinarySize, client->fId);
         client->fWebWindow->SendBinary(client->fId, &fOutputBinary[0], fTotalBinarySize);
      }
   }
}


/*
////////////////////////////////////////////////////////////////////////////////
/// Repaint the scene.

void TEveScene::Repaint(Bool_t dropLogicals)
{
   if (dropLogicals) fGLScene->SetSmartRefresh(kFALSE);
   fGLScene->PadPaint(fPad);
   if (dropLogicals) fGLScene->SetSmartRefresh(kTRUE);
   fChanged = kFALSE;

   // Hack to propagate selection state to physical shapes.
   //
   // Should actually be published in PadPaint() following a direct
   // AddObject() call, but would need some other stuff for that.
   // Optionally, this could be exported via the TAtt3D and everything
   // would be sweet.

   TGLScene::LogicalShapeMap_t& logs = fGLScene->RefLogicalShapes();
   TEveElement* elm;
   for (TGLScene::LogicalShapeMapIt_t li = logs.begin(); li != logs.end(); ++li)
   {
      elm = dynamic_cast<TEveElement*>(li->first);
      if (elm && li->second->Ref() == 1)
      {
         TGLPhysicalShape* pshp = const_cast<TGLPhysicalShape*>(li->second->GetFirstPhysical());
         pshp->Select(elm->GetSelectedLevel());
      }
   }

   // Fix positions for hierarchical scenes.
   if (fHierarchical)
   {
      RetransHierarchically();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Entry point for hierarchical transformation update.
/// Calls the recursive variant on all children.

void TEveScene::RetransHierarchically()
{
   fGLScene->BeginUpdate();

   RetransHierarchicallyRecurse(this, RefMainTrans());

   fGLScene->EndUpdate();
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix for physical shape of element el in
/// the GL-scene and recursively descend into children (if enabled).

void TEveScene::RetransHierarchicallyRecurse(TEveElement* el, const TEveTrans& tp)
{
   static const TEveException eh("TEveScene::RetransHierarchicallyRecurse ");

   TEveTrans t(tp);
   if (el->HasMainTrans())
      t *= el->RefMainTrans();

   if (el->GetRnrSelf() && el != this)
   {
      fGLScene->UpdatePhysioLogical(el->GetRenderObject(eh), t.Array(), 0);
   }

   if (el->GetRnrChildren())
   {
      for (List_i i = el->BeginChildren(); i != el->EndChildren(); ++i)
      {
         if ((*i)->GetRnrAnything())
            RetransHierarchicallyRecurse(*i, t);
      }
   }
}
*/


/*
////////////////////////////////////////////////////////////////////////////////
/// Paint the scene. Iterate over children and calls PadPaint().

void TEveScene::Paint(Option_t* option)
{
   if (GetRnrState())
   {
      for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
      {
         // (*i)->PadPaint(option);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from the scene.
/// It is not an error if the element is not found in the scene.

void TEveScene::DestroyElementRenderers(TEveElement* element)
{
   static const TEveException eh("TEveScene::DestroyElementRenderers ");

   fGLScene->BeginUpdate();
   Bool_t changed = fGLScene->DestroyLogical(element->GetRenderObject(eh), kFALSE);
   fGLScene->EndUpdate(changed, changed);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element represented by object rnrObj from the scene.
/// It is not an error if the element is not found in the scene.

void TEveScene::DestroyElementRenderers(TObject* rnrObj)
{
   fGLScene->BeginUpdate();
   Bool_t changed = fGLScene->DestroyLogical(rnrObj, kFALSE);
   fGLScene->EndUpdate(changed, changed);
}
*/


/** \class TEveSceneList
\ingroup TEve
List of Scenes providing common operations on TEveScene collections.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveSceneList::TEveSceneList(const char* n, const char* t) :
   TEveElementList(n, t)
{
   SetChildClass(TEveScene::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy all scenes and their contents.
/// Tho object with non-zero deny-destroy will still survive.

void TEveSceneList::DestroyScenes()
{
   List_i i = fChildren.begin();
   while (i != fChildren.end())
   {
      TEveScene* s = (TEveScene*) *(i++);
      s->DestroyElements();
      s->DestroyOrWarn();
   }
}

/*
////////////////////////////////////////////////////////////////////////////////
/// Repaint scenes that are tagged as changed.

void TEveSceneList::RepaintChangedScenes(Bool_t dropLogicals)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      if (s->IsChanged())
      {
         s->Repaint(dropLogicals);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Repaint all scenes.

void TEveSceneList::RepaintAllScenes(Bool_t dropLogicals)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      ((TEveScene*) *i)->Repaint(dropLogicals);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Loop over all scenes and remove all instances of element from them.

void TEveSceneList::DestroyElementRenderers(TEveElement* element)
{
   static const TEveException eh("TEveSceneList::DestroyElementRenderers ");

   TObject* obj = element->GetRenderObject(eh);
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      ((TEveScene*)*i)->DestroyElementRenderers(obj);
   }
}

*/

////////////////////////////////////////////////////////////////////////////////
//
// Send an update of element representations
//
////////////////////////////////////////////////////////////////////////////////

void TEveSceneList::ProcessSceneChanges()
{
   printf("ProcessSceneChanges\n");

   for (List_i sIt=fChildren.begin(); sIt!=fChildren.end(); ++sIt)
   {
      TEveScene* s = (TEveScene*) *sIt;
      s->StreamRepresentationChanges();
      s->SendChangesToSubscribers();
   }
}
