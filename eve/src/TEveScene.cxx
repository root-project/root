// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveScene.h>
#include <TEveViewer.h>
#include <TEveManager.h>

#include <TList.h>
#include <TGLScenePad.h>

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
   fPad = new TEvePad;
   fPad->GetListOfPrimitives()->Add(this);
   fGLScene = new TGLScenePad(fPad);
   fGLScene->SetName(n);
   fGLScene->SetAutoDestruct(kFALSE);
}

//______________________________________________________________________________
TEveScene::~TEveScene()
{
   gEve->GetViewers()->SceneDestructing(this);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::CollectSceneParents(List_t& scenes)
{
   scenes.push_back(this);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::Repaint()
{
   fGLScene->PadPaint(fPad);
   fChanged = kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveScene::SetName(const Text_t* n)
{
   TEveElementList::SetName(n);
   fGLScene->SetName(n);
}

//______________________________________________________________________________
void TEveScene::Paint(Option_t* option)
{
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
   SetChildClass(TEveScene::Class());
}

//______________________________________________________________________________
TEveSceneList::~TEveSceneList()
{}

/******************************************************************************/

//______________________________________________________________________________
void TEveSceneList::RepaintChangedScenes()
{
   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      if (s->IsChanged())
      {
         // printf(" TEveScene '%s' changed ... repainting.\n", s->GetName());
         s->Repaint();
      }
   }
}

//______________________________________________________________________________
void TEveSceneList::RepaintAllScenes()
{
   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveScene* s = (TEveScene*) *i;
      // printf(" TEveScene '%s' repainting.\n", s->GetName());
      s->Repaint();
   }
}
