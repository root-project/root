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

#include "TList.h"
#include "TGLScenePad.h"

//______________________________________________________________________________
// TEveScene
//
// Reve representation of TGLScene.

ClassImp(TEveScene)

//______________________________________________________________________________
TEveScene::TEveScene(const Text_t* n, const Text_t* t) :
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

   gEve->GetViewers()->SceneDestructing(this);
}

/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveScene::GetListTreeIcon() 
{ 
   //return evescene icon
   return TEveElement::fgListTreeIcons[2]; 
}

//______________________________________________________________________________
void TEveScene::CollectSceneParents(List_t& scenes)
{
   // Virtual from TEveElement; here we simply append this scene to
   // the list.

   scenes.push_back(this);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::Repaint()
{
   // Repaint the scene.

   fGLScene->PadPaint(fPad);
   fChanged = kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::SetName(const Text_t* n)
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


//______________________________________________________________________________
// TEveSceneList
//
// List of Scenes providing common operations on TEveScene collections.

ClassImp(TEveSceneList)

//______________________________________________________________________________
TEveSceneList::TEveSceneList(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t)
{
   // Constructor.

   SetChildClass(TEveScene::Class());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveSceneList::RepaintChangedScenes(Bool_t dropLogicals)
{
   // Repaint scenes that are tagged as changed.

   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      if (s->IsChanged())
      {
         if (dropLogicals) s->GetGLScene()->SetSmartRefresh(kFALSE);
         s->Repaint();
         if (dropLogicals) s->GetGLScene()->SetSmartRefresh(kTRUE);
      }
   }
}

//______________________________________________________________________________
void TEveSceneList::RepaintAllScenes(Bool_t dropLogicals)
{
   // Repaint all scenes.

   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      if (dropLogicals) s->GetGLScene()->SetSmartRefresh(kFALSE);
      s->Repaint();
      if (dropLogicals) s->GetGLScene()->SetSmartRefresh(kTRUE);
   }
}
