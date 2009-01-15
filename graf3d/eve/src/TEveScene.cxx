// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveTrans.h"

#include "TList.h"
#include "TGLScenePad.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"

//==============================================================================
//==============================================================================
// TEveScene
//==============================================================================

//______________________________________________________________________________
//
// Eve representation of TGLScene.
// The GLScene is owned by this class - it is created on construction
// time and deleted at destruction.

ClassImp(TEveScene);

//______________________________________________________________________________
TEveScene::TEveScene(const char* n, const char* t) :
   TEveElementList(n, t),
   fPad    (0),
   fGLScene(0),
   fChanged      (kFALSE),
   fSmartRefresh (kTRUE)
{
   // Constructor.

   fPad = new TEvePad;
   fPad->GetListOfPrimitives()->Add(this);
   fGLScene = new TGLScenePad(fPad);
   fGLScene->SetName(n);
   fGLScene->SetAutoDestruct(kFALSE);
   fGLScene->SetSmartRefresh(kTRUE);
}

//______________________________________________________________________________
TEveScene::~TEveScene()
{
   // Destructor.

   fDestructing = kTRUE;

   gEve->GetViewers()->SceneDestructing(this);
   gEve->GetScenes()->RemoveElement(this);
   delete fGLScene;
   delete fPad;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::CollectSceneParents(List_t& scenes)
{
   // Virtual from TEveElement; here we simply append this scene to
   // the list.

   scenes.push_back(this);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::Repaint(Bool_t dropLogicals)
{
   // Repaint the scene.

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
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::SetName(const char* n)
{
   // Set scene's name.

   TEveElementList::SetName(n);
   fGLScene->SetName(n);
}

//______________________________________________________________________________
void TEveScene::Paint(Option_t* option)
{
   // Paint the scene. Iterate over children and calls PadPaint().

   if (fRnrChildren)
   {
      for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
         (*i)->PadPaint(option);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::DestroyElementRenderers(TEveElement* element)
{
   // Remove element from the scene.
   // It is not an error if the element is not found in the scene.

   fGLScene->BeginUpdate();
   Bool_t changed = fGLScene->DestroyLogical(element->GetRenderObject(), kFALSE);
   fGLScene->EndUpdate(changed, changed);
}

//______________________________________________________________________________
void TEveScene::DestroyElementRenderers(TObject* rnrObj)
{
   // Remove element represented by object rnrObj from the scene.
   // It is not an error if the element is not found in the scene.

   fGLScene->BeginUpdate();
   Bool_t changed = fGLScene->DestroyLogical(rnrObj, kFALSE);
   fGLScene->EndUpdate(changed, changed);
}

/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveScene::GetListTreeIcon(Bool_t)
{
   // Return icon for scene.

   return TEveElement::fgListTreeIcons[2];
}


//==============================================================================
//==============================================================================
// TEveSceneList
//==============================================================================

//______________________________________________________________________________
//
// List of Scenes providing common operations on TEveScene collections.

ClassImp(TEveSceneList);

//______________________________________________________________________________
TEveSceneList::TEveSceneList(const char* n, const char* t) :
   TEveElementList(n, t)
{
   // Constructor.

   SetChildClass(TEveScene::Class());
}

//______________________________________________________________________________
void TEveSceneList::DestroyScenes()
{
   // Destroy all scenes and their contents.
   // Tho object with non-zero deny-destroy will still survive.

   List_i i = fChildren.begin();
   while (i != fChildren.end())
   {
      TEveScene* s = (TEveScene*) *(i++);
      s->DestroyElements();
      s->DestroyOrWarn();
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveSceneList::RepaintChangedScenes(Bool_t dropLogicals)
{
   // Repaint scenes that are tagged as changed.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      if (s->IsChanged())
      {
         s->Repaint(dropLogicals);
      }
   }
}

//______________________________________________________________________________
void TEveSceneList::RepaintAllScenes(Bool_t dropLogicals)
{
   // Repaint all scenes.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      ((TEveScene*) *i)->Repaint(dropLogicals);
   }
}

//______________________________________________________________________________
void TEveSceneList::DestroyElementRenderers(TEveElement* element)
{
   // Loop over all scenes and remove all instances of element from
   // them.

   TObject* obj = element->GetRenderObject();
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      ((TEveScene*)*i)->DestroyElementRenderers(obj);
   }
}

//______________________________________________________________________________
void TEveSceneList::ProcessSceneChanges(Bool_t dropLogicals, Set_t& stampSet)
{
   // Loop over all scenes and update them accordingly:
   //   a) if scene is marked as changed, it is repainted;
   //   b) otherwise iteration is done over the set of stamped elements and
   //      their physical/logical shapes are updated accordingly.
   //
   // This allows much finer update granularity without resetting of
   // complex GL-viewer and GL-scene state.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;

      if (s->IsChanged())
      {
         s->Repaint(dropLogicals);
      }
      else
      {
         Bool_t updateViewers = kFALSE;
         Bool_t incTimeStamp  = kFALSE;

         s->GetGLScene()->BeginUpdate();

         // Process stamps.
         TGLScene::LogicalShapeMap_t& logs = s->GetGLScene()->RefLogicalShapes();
         TGLScene::LogicalShapeMapIt_t li = logs.begin();

         Set_i ei = stampSet.begin();

         TObject* eobj = 0;

         while (li != logs.end() && ei != stampSet.end())
         {
            if (!eobj) eobj = (*ei)->GetRenderObject();

            if (li->first == eobj)
            {
               if (li->second->Ref() != 1)
                  Warning("TEveSceneList::ProcessSceneChanges",
                          "Expect one physical, cnt=%u.", li->second->Ref());

               TGLLogicalShape  *lshp = li->second;
               TGLPhysicalShape *pshp = const_cast<TGLPhysicalShape*>(lshp->GetFirstPhysical());
               TEveElement      *el   = *ei;
               UChar_t           bits = el->GetChangeBits();

               if (bits & kCBColorSelection)
               {
                  pshp->Select(el->GetSelectedLevel());
                  pshp->SetDiffuseColor(el->GetMainColor(),
                                        el->GetMainTransparency());
               }

               if (bits & kCBTransBBox)
               {
                  if (el->HasMainTrans())
                     pshp->SetTransform(el->PtrMainTrans()->Array());
                  lshp->UpdateBoundingBox();
                  incTimeStamp = kTRUE;
               }

               if (bits & kCBObjProps)
               {
                  lshp->DLCacheClear();
               }

               ++li; ++ei; eobj = 0;
               updateViewers = kTRUE;
            }
            else if (li->first < eobj)
            {
               ++li;
            }
            else
            {
               ++ei; eobj = 0;
            }
         }

         s->GetGLScene()->EndUpdate(updateViewers, incTimeStamp, updateViewers);
      }
   }
}
