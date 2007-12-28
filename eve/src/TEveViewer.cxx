// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveViewer.h"
#include "TEveScene.h"
#include "TEveSceneInfo.h"

#include "TEveManager.h"

#include "TGLSAViewer.h"
#include "TGLScenePad.h"

#include "TGLOrthoCamera.h" // For fixing defaults in root 5.17.4

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
   // Constructor.

   SetChildClass(TEveSceneInfo::Class());
}

/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveViewer::GetListTreeIcon() 
{ 
   //return eveviewer icon
   return TEveElement::fgListTreeIcons[1]; 
}

//______________________________________________________________________________
void TEveViewer::SetGLViewer(TGLViewer* s)
{
   // Set TGLViewer that is represented by this object.

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
   // Spawn new GLViewer and adopt it.

   TGLSAViewer* v = new TGLSAViewer(parent, 0, ged);
   v->ToggleEditObject();
   SetGLViewer(v);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewer::AddScene(TEveScene* scene)
{
   // Add 'scene' to the list of scenes.

   static const TEveException eh("TEveViewer::AddScene ");

   TGLSceneInfo* glsi = fGLViewer->AddScene(scene->GetGLScene());
   if (glsi != 0) {
      TEveSceneInfo* si = new TEveSceneInfo(this, scene, glsi);
      gEve->AddElement(si, this);
   } else {
      throw(eh + "scene already in the viewer.");
   }
}

//______________________________________________________________________________
void TEveViewer::RemoveElementLocal(TEveElement* el)
{
   // Remove element 'el' from the list of children and also remove
   // appropriate GLScene from GLViewer's list of scenes.
   // Virtual from TEveElement.

   fGLViewer->RemoveScene(((TEveSceneInfo*)el)->GetGLScene());
}

//______________________________________________________________________________
void TEveViewer::RemoveElementsLocal()
{
   // Remove all children, forwarded to GLViewer.
   // Virtual from TEveElement.

   fGLViewer->RemoveAllScenes();
}

//______________________________________________________________________________
TObject* TEveViewer::GetEditorObject(const TEveException& eh) const
{
   // Object to be edited when this is selected, returns the TGLViewer.
   // Virtual from TEveElement.

   if (!fGLViewer)
      throw(eh + "fGLViewer not set.");
   return fGLViewer;
}

//______________________________________________________________________________
Bool_t TEveViewer::HandleElementPaste(TEveElement* el)
{
   // Receive a pasted object. TEveViewer only accepts objects of
   // class TEveScene.
   // Virtual from TEveElement.

   static const TEveException eh("TEveViewer::HandleElementPaste ");

   TEveScene* scene = dynamic_cast<TEveScene*>(el);
   if (scene != 0) {
      AddScene(scene);
      return kTRUE;
   } else {
      Warning(eh.Data(), "class TEveViewer only accepts TEveScene paste argument.");
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
   // Constructor.

   SetChildClass(TEveViewer::Class());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals)
{
   // Repaint viewers that are tagged as changed.

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
   // Repaint all viewers.

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
   // Callback done from a TEveScene destructor allowing proper
   // removal of the scene from affected viewers.

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
