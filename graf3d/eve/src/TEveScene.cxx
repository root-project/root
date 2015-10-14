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
#include "TEvePad.h"

#include "TGLScenePad.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"

#include "TList.h"
#include "TExMap.h"

/** \class TEveScene
\ingroup TEve
Eve representation of TGLScene.
The GLScene is owned by this class - it is created on construction
time and deleted at destruction.

Normally all objects are positioned directly in global scene-space.
By setting the fHierarchical flag, positions of children get
calculated by multiplying the transformation matrices of all parents.
*/

ClassImp(TEveScene);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveScene::TEveScene(const char* n, const char* t) :
   TEveElementList(n, t),
   fPad    (0),
   fGLScene(0),
   fChanged      (kFALSE),
   fSmartRefresh (kTRUE),
   fHierarchical (kFALSE)
{
   fPad = new TEvePad;
   fPad->GetListOfPrimitives()->Add(this);
   fGLScene = new TGLScenePad(fPad);
   fGLScene->SetName(n);
   fGLScene->SetAutoDestruct(kFALSE);
   fGLScene->SetSmartRefresh(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveScene::TEveScene(TGLScenePad* gl_scene, const char* n, const char* t) :
   TEveElementList(n, t),
   fPad    (0),
   fGLScene(gl_scene),
   fChanged      (kFALSE),
   fSmartRefresh (kTRUE),
   fHierarchical (kFALSE)
{
   fPad = new TEvePad;
   fPad->GetListOfPrimitives()->Add(this);
   fGLScene->SetPad(fPad);
   fGLScene->SetName(n);
   fGLScene->SetAutoDestruct(kFALSE);
   fGLScene->SetSmartRefresh(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveScene::~TEveScene()
{
   fDestructing = kStandard;

   gEve->GetViewers()->SceneDestructing(this);
   gEve->GetScenes()->RemoveElement(this);
   delete fGLScene;
   delete fPad;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement; here we simply append this scene to
/// the list.

void TEveScene::CollectSceneParents(List_t& scenes)
{
   scenes.push_back(this);
}

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

////////////////////////////////////////////////////////////////////////////////
/// Set scene's name.

void TEveScene::SetName(const char* n)
{
   TEveElementList::SetName(n);
   fGLScene->SetName(n);
}

////////////////////////////////////////////////////////////////////////////////
/// Paint the scene. Iterate over children and calls PadPaint().

void TEveScene::Paint(Option_t* option)
{
   if (GetRnrState())
   {
      for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
         (*i)->PadPaint(option);
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

////////////////////////////////////////////////////////////////////////////////
/// Return icon for scene.

const TGPicture* TEveScene::GetListTreeIcon(Bool_t)
{
   return TEveElement::fgListTreeIcons[2];
}

/** \class TEveSceneList
\ingroup TEve
List of Scenes providing common operations on TEveScene collections.
*/

ClassImp(TEveSceneList);

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

////////////////////////////////////////////////////////////////////////////////
/// Loop over all scenes and update them accordingly:
///  1. if scene is marked as changed, it is repainted;
///  2. otherwise iteration is done over the set of stamped elements and
///     their physical/logical shapes are updated accordingly.
///
/// This allows much finer update granularity without resetting of
/// complex GL-viewer and GL-scene state.

void TEveSceneList::ProcessSceneChanges(Bool_t dropLogicals, TExMap* stampMap)
{
   // We need changed elements sorted by their "render object" as we do
   // parallel iteration over this list and the list of logical shapes
   // in every scene.

   static const TEveException eh("TEveSceneList::ProcessSceneChanges ");

   typedef std::map<TObject*, TEveElement*> mObjectElement_t;
   typedef mObjectElement_t::iterator       mObjectElement_i;

   mObjectElement_t changed_objects;
   {
      Long64_t   key, value;
      TExMapIter stamped_elements(stampMap);
      while (stamped_elements.Next(key, value))
      {
         TEveElement *el = reinterpret_cast<TEveElement*>(key);
         changed_objects.insert(std::make_pair(el->GetRenderObject(eh), el));
      }
   }

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
         Bool_t transbboxChg  = kFALSE;

         s->GetGLScene()->BeginUpdate();

         // Process stamps.
         TGLScene::LogicalShapeMap_t   &logs = s->GetGLScene()->RefLogicalShapes();
         TGLScene::LogicalShapeMapIt_t  li   = logs.begin();

         mObjectElement_i ei = changed_objects.begin();

         while (li != logs.end() && ei != changed_objects.end())
         {
            if (li->first == ei->first)
            {
               if (li->second->Ref() != 1)
                  Warning("TEveSceneList::ProcessSceneChanges",
                          "Expect one physical, cnt=%u.", li->second->Ref());

               TGLLogicalShape  *lshp = li->second;
               TGLPhysicalShape *pshp = const_cast<TGLPhysicalShape*>(lshp->GetFirstPhysical());
               TEveElement      *el   = ei->second;
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
                  transbboxChg = kTRUE;
               }

               if (bits & kCBObjProps)
               {
                  lshp->DLCacheClear();
               }

               ++li; ++ei;
               updateViewers = kTRUE;
            }
            else if (li->first < ei->first)
            {
               ++li;
            }
            else
            {
               ++ei;
            }
         }

         s->GetGLScene()->EndUpdate(updateViewers, incTimeStamp, updateViewers);

         // Fix positions for hierarchical scenes.
         if (s->GetHierarchical() && transbboxChg)
         {
            s->RetransHierarchically();
         }
      }
   }
}
