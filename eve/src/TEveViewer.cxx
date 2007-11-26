// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveViewer.h>
#include <TEveScene.h>
#include <TEveSceneInfo.h>

#include <TEveManager.h>

#include <TGLSAViewer.h>
#include <TGLScenePad.h>

#include <TGLOrthoCamera.h> // For fixing defaults in root 5.17.4

//______________________________________________________________________________
// TEveViewer
//
// Reve representation of TGLViewer.

ClassImp(TEveViewer)

//______________________________________________________________________________
TEveViewer::TEveViewer(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t),
   fGLViewer (0)
{
   SetChildClass(TEveSceneInfo::Class());
}

//______________________________________________________________________________
TEveViewer::~TEveViewer()
{}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewer::SetGLViewer(TGLViewer* s)
{
   delete fGLViewer;
   fGLViewer = s;

   fGLViewer->SetSmartRefresh(kTRUE);
   // fGLViewer->SetResetCamerasOnUpdate(kFALSE);
   fGLViewer->SetResetCameraOnDoubleClick(kFALSE);

   // Temporary fix for wrong defaults in root 5.17.04
   fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ((TGLOrthoCamera&)(fGLViewer->CurrentCamera())).SetEnableRotate(kTRUE);
   fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoXOZ);
   ((TGLOrthoCamera&)(fGLViewer->CurrentCamera())).SetEnableRotate(kTRUE);
   fGLViewer->SetCurrentCamera(TGLViewer::kCameraOrthoZOY);
   ((TGLOrthoCamera&)(fGLViewer->CurrentCamera())).SetEnableRotate(kTRUE);
   fGLViewer->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
}

//______________________________________________________________________________
void TEveViewer::SpawnGLViewer(const TGWindow* parent, TGedEditor* ged)
{
   TGLSAViewer* v = new TGLSAViewer(parent, 0, ged);
   v->ToggleEditObject();
   SetGLViewer(v);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewer::AddScene(TEveScene* scene)
{
   static const TEveException eH("TEveViewer::AddScene ");

   TGLSceneInfo* glsi = fGLViewer->AddScene(scene->GetGLScene());
   if (glsi != 0) {
      TEveSceneInfo* si = new TEveSceneInfo(this, scene, glsi);
      gEve->AddElement(si, this);
   } else {
      throw(eH + "scene already in the viewer.");
   }
}

//______________________________________________________________________________
void TEveViewer::RemoveElementLocal(TEveElement* el)
{
   fGLViewer->RemoveScene(((TEveSceneInfo*)el)->GetGLScene());
}

//______________________________________________________________________________
void TEveViewer::RemoveElementsLocal()
{
   fGLViewer->RemoveAllScenes();
}

//______________________________________________________________________________
TObject* TEveViewer::GetEditorObject() const
{
   return fGLViewer;
}

//______________________________________________________________________________
Bool_t TEveViewer::HandleElementPaste(TEveElement* el)
{
   static const TEveException eH("TEveViewer::HandleElementPaste ");

   TEveScene* scene = dynamic_cast<TEveScene*>(el);
   if (scene != 0) {
      AddScene(scene);
      return kTRUE;
   } else {
      Warning(eH.Data(), "class TEveViewer only accepts TEveScene paste argument.");
      return kFALSE;
   }
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

//______________________________________________________________________________
// TEveViewerList
//
// List of Viewers providing common operations on TEveViewer collections.

ClassImp(TEveViewerList)

//______________________________________________________________________________
TEveViewerList::TEveViewerList(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t)
{
   SetChildClass(TEveViewer::Class());
}

//______________________________________________________________________________
TEveViewerList::~TEveViewerList()
{}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      if (glv->IsChanged())
      {
         // printf(" TEveViewer '%s' changed ... reqesting draw.\n", (*i)->GetObject()->GetName());

         if (resetCameras)        glv->PostSceneBuildSetup(kTRUE);
         if (dropLogicals) glv->SetSmartRefresh(kFALSE);

         glv->RequestDraw(TGLRnrCtx::kLODHigh);

         if (dropLogicals) glv->SetSmartRefresh(kTRUE);
      }
   }
}

//______________________________________________________________________________
void TEveViewerList::RepaintAllViewers(Bool_t resetCameras, Bool_t dropLogicals)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();

      // printf(" TEveViewer '%s' sending redraw reqest.\n", (*i)->GetObject()->GetName());

      if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      glv->RequestDraw(TGLRnrCtx::kLODHigh);

      if (dropLogicals) glv->SetSmartRefresh(kTRUE);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::SceneDestructing(TEveScene* scene)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveViewer* viewer = (TEveViewer*) *i;
      List_i j = viewer->BeginChildren();
      while (j != viewer->EndChildren())
      {
         TEveSceneInfo* sinfo = (TEveSceneInfo*) *j;
         ++j;
         if (sinfo->GetScene() == scene)
            gEve->RemoveElement(sinfo, viewer);
      }
   }
}
