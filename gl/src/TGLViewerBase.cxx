// @(#)root/gl:$Name:  $:$Id: TGLViewerBase.cxx,v 1.3 2007/06/22 15:11:13 brun Exp $
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLViewerBase.h"

#include "TGLSceneBase.h"
#include "TGLSceneInfo.h"

#include <TGLRnrCtx.h>
#include "TGLCamera.h"
#include <TGLOverlay.h>
#include <TGLSelectBuffer.h>
#include <TGLSelectRecord.h>
#include <TGLUtil.h>
#include "TGLContext.h"
#include "TGLIncludes.h"

#include <stdexcept>

//______________________________________________________________________
// TGLViewerBase
//
// Base class for GL viewers. Provides a basic scene management and a
// small set of control variables (camera, LOD, style, clip) that are
// used by the scene classes. Renering wrappers are available but
// minimal.
//
// There is no concept of GL-context here ... we just draw
// into whatever is set from outside.
//
// Development notes:
//
// Each viewer automatically creates a TGLRnrCtx and passes it down
// all render functions.

ClassImp(TGLViewerBase)

//______________________________________________________________________
TGLViewerBase::TGLViewerBase() :
   fRnrCtx    (0),
   fCamera    (0),
   fClip      (0),
   fLOD       (TGLRnrCtx::kLODHigh),
   fStyle     (TGLRnrCtx::kFill),

   fResetSceneInfosOnRender (kFALSE)
{
   // Constructor.

   fRnrCtx = new TGLRnrCtx(this);
}

//______________________________________________________________________
TGLViewerBase::~TGLViewerBase()
{
   // Destructor.

   delete fRnrCtx;
}

//______________________________________________________________________________
const char* TGLViewerBase::LockIdStr() const
{
   // Name to print in locking output.

   return "TGLViewerBase";
}

/**************************************************************************/
// Scene & scene-info management
/**************************************************************************/

//______________________________________________________________________
TGLViewerBase::SceneInfoList_i
TGLViewerBase::FindScene(TGLSceneBase* scene)
{
   // Find scene-info corresponding to scene.

   SceneInfoList_i i = fScenes.begin();
   while (i != fScenes.end() && (*i)->GetScene() != scene) ++i;
   return i;
}

//______________________________________________________________________
void TGLViewerBase::AddScene(TGLSceneBase* scene)
{
   // Add new scene, appropriate scene-info is created.

   fScenes.push_back(scene->CreateSceneInfo(this));
}

//______________________________________________________________________
void TGLViewerBase::RemoveScene(TGLSceneBase* scene)
{
   // Remove scene, its scene-info is deleted.

   SceneInfoList_i i = FindScene(scene);
   if (i != fScenes.end()) {
      delete *i;
      fScenes.erase(i);
   } else {
      Warning("TGLViewerBase::RemoveScene", "scene not found.");
   }
}

//______________________________________________________________________
TGLSceneInfo* TGLViewerBase::GetSceneInfo(TGLSceneBase* scene)
{
   // Find scene-info corresponding to scene.

   SceneInfoList_i i = FindScene(scene);
   if (i != fScenes.end())
      return *i;
   else
      return 0;
}


/**************************************************************************/
// SceneInfo update / check
/**************************************************************************/

void TGLViewerBase::ResetSceneInfos()
{
   // Force rebuild of view-dependent scene-info structures.
   //
   // This should be called before calling render (draw/select) if
   // something that affects rendering has been changed.
   //
   // We now use timestamps for clip / camera, so this should rarely
   // be needed.

   SceneInfoList_i i = fScenes.begin();
   while (i != fScenes.end())
   {
      (*i)->ResetSceneStamp();
      ++i;
   }
}


/**************************************************************************/
// Rendering / selection virtuals
/**************************************************************************/

//______________________________________________________________________
void TGLViewerBase::PreRender()
{
   // Initialize render-context, setup camera, GL, render-area.
   // Check and lock scenes, determine their visibility.

   TGLContextIdentity* cid = TGLContextIdentity::GetCurrent();
   if (cid == 0)
   {
      // Assume derived class set it up for us.
      // This happens due to very complex and involved implementation
      // of gl-in-pad that uses gGLManager.
      // In principle we should throw an exception:
      // throw std::runtime_error("Can not resolve GL context.");
   }
   else
   {
      if (cid != fRnrCtx->GetGLCtxIdentity())
      {
         if (fRnrCtx->GetGLCtxIdentity() != 0)
            Warning("TGLViewerBase::PreRender", "Switching to another GL context; maybe you should use context-sharing.");
         fRnrCtx->SetGLCtxIdentity(cid);      
      }
   }

   fRnrCtx->SetCamera      (fCamera);
   fRnrCtx->SetViewerLOD   (fLOD);
   fRnrCtx->SetViewerStyle (fStyle);
   fRnrCtx->SetViewerClip  (fClip);

   if (fResetSceneInfosOnRender)
   {
      ResetSceneInfos();
      fResetSceneInfosOnRender = kFALSE;
   }

   fOverallBoundingBox.SetEmpty();
   SceneInfoList_t locked_scenes;
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      TGLSceneInfo * sinfo = *i;
      if ( ! sinfo->GetScene()->TakeLock(kDrawLock))
      {
         Warning("TGLViewerBase::PreRender", "locking of scene '%s' failed, skipping.",
                 sinfo->GetScene()->GetName());
         continue;
      }
      sinfo->SetupTransformsAndBBox(); // !!! transform not done yet
      fOverallBoundingBox.MergeAligned(sinfo->GetTransformedBBox());
      locked_scenes.push_back(sinfo);
   }

   fCamera->Apply(fOverallBoundingBox, fRnrCtx->GetPickRectangle());

   // Make precursory selection of visible scenes.
   // Only scene bounding-box .vs. camera frustum check performed.
   fVisScenes.clear();
   for (SceneInfoList_i i=locked_scenes.begin(); i!=locked_scenes.end(); ++i)
   {
      TGLSceneInfo         * sinfo = *i;
      const TGLBoundingBox & bbox  = sinfo->GetTransformedBBox();
      Bool_t visp = (!bbox.IsEmpty() && fCamera->FrustumOverlap(bbox) != kOutside);
      sinfo->ViewCheck(visp);
      if (visp)
         fVisScenes.push_back(sinfo);
      else
         sinfo->GetScene()->ReleaseLock(kDrawLock);
   }
}

//______________________________________________________________________
void TGLViewerBase::Render()
{
   // Go through a list of scenes and render them in order.

   // !!! should be split into two passes: opaque / transparent.

   Int_t nScenes = fVisScenes.size();
   for (Int_t i = 0; i < nScenes; ++i)
   {
      TGLSceneInfo* sinfo = fVisScenes[i];
      TGLSceneBase* scene = sinfo->GetScene();
      fRnrCtx->SetSceneInfo(sinfo);
      glPushName(i);
      scene->FullRender(*fRnrCtx);
      glPopName();
      fRnrCtx->SetSceneInfo(0);
   }

   TGLUtil::CheckError("TGLViewerBase::Render - pre exit check");
}

//______________________________________________________________________
void TGLViewerBase::RenderOverlay()
{
   // Render overlay objects.

   Int_t nOvl = fOverlay.size();
   for (Int_t i = 0; i < nOvl; ++i)
   {
      TGLOverlayElement* el = fOverlay[i];
      glPushName(i);
      el->Render(*fRnrCtx);
      glPopName();
   }
}

//______________________________________________________________________
void TGLViewerBase::PostRender()
{
   // Function called after rendering is finished.
   // Here we just unlock the scenes.

   Int_t nScenes = fVisScenes.size();
   for (Int_t i = 0; i < nScenes; ++i)
   {
      fVisScenes[i]->GetScene()->ReleaseLock(kDrawLock);
   }
}

//______________________________________________________________________
void TGLViewerBase::PreRenderOverlaySelection()
{
   // Perform minimal initialization for overlay selection.
   // Here we assume that scene has already been drawn and that
   // camera and overall bounding box are ok.
   // Scenes are not locked.

   fCamera->Apply(fOverallBoundingBox, fRnrCtx->GetPickRectangle());
}

//______________________________________________________________________
void TGLViewerBase::PostRenderOverlaySelection()
{
   // Perform cleanup after overlay selection.

}

/**************************************************************************/
// High-level functions: drawing and picking.
/**************************************************************************/


//______________________________________________________________________
//void TGLViewerBase::Select(Int_t selX, Int_t selY, Int_t selRadius)
//{
   // Perform render-pass in selection mode.
   // Process the selection results.
   // For now only in derived classes.
//}

//______________________________________________________________________
Bool_t TGLViewerBase::ResolveSelectRecord(TGLSelectRecord& rec, Int_t recIdx)
{
   // Process selection record on buffer-position 'recIdx' and
   // fill the data into 'rec'.
   //
   // Returns TRUE if scene was demangled and an object identified.
   // When FALSE is returned it is still possible that scene has been
   // identified. Check for this if interested in scene-selection.
   //
   // The select-buffer is taken form fRnrCtx.

   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();
   if (recIdx >= sb->GetNRecords())
       return kFALSE;

   sb->SelectRecord(rec, recIdx);
   UInt_t sceneIdx = rec.GetItem(0);
   if (sceneIdx >= fVisScenes.size())
       return kFALSE;

   TGLSceneInfo* sinfo = fVisScenes[sceneIdx];
   rec.SetSceneInfo(sinfo);
   return sinfo->GetScene()->ResolveSelectRecord(rec, 1);
}

//______________________________________________________________________
Bool_t TGLViewerBase::FindClosestRecord(TGLSelectRecord& rec, Int_t& recIdx)
{
   // Find next select record that can be resolved, starting from
   // position 'recIdx'.
   // 'recIdx' is passed as reference and points to found record in the buffer.

   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();

   while (recIdx < sb->GetNRecords())
   {
      if (ResolveSelectRecord(rec, recIdx))
         return kTRUE;
      ++recIdx;
   }
   return kFALSE;
}

//______________________________________________________________________
Bool_t TGLViewerBase::FindClosestOpaqueRecord(TGLSelectRecord& rec, Int_t& recIdx)
{
   // Find next select record that can be resolved and whose result is
   // not transparent, starting from position 'recIdx'.
   // 'recIdx' is passed as reference and points to found record in the buffer.

   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();

   while (recIdx < sb->GetNRecords())
   {
      if (ResolveSelectRecord(rec, recIdx) && ! rec.GetTransparent())
         return kTRUE;
      ++recIdx;
   }
   return kFALSE;
}

//______________________________________________________________________
Bool_t TGLViewerBase::FindClosestOverlayRecord(TGLOvlSelectRecord& rec,
                                               Int_t             & recIdx)
{
   // Find next overlay-select record that can be resolved, starting from
   // position 'recIdx'.
   // 'recIdx' is passed as reference and points to found record in the buffer.

   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();

   while (recIdx < sb->GetNRecords())
   {
      sb->SelectRecord(rec, recIdx);
      if (rec.GetItem(0) < fOverlay.size())
      {
         rec.SetOvlElement(fOverlay[rec.GetItem(0)]);
         rec.NextPos();
         return kTRUE;
      }
      ++recIdx;
   }
   return kFALSE;
}
