// @(#)root/gl:$Id$
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

#include "TGLRnrCtx.h"
#include "TGLCamera.h"
#include "TGLClip.h"
#include "TGLOverlay.h"
#include "TGLSelectBuffer.h"
#include "TGLSelectRecord.h"
#include "TGLAnnotation.h"
#include "TGLUtil.h"

#include "TGLContext.h"
#include "TGLIncludes.h"

#include "TEnv.h"

#include <algorithm>
#include <stdexcept>

/** \class TGLViewerBase
\ingroup opengl
Base class for GL viewers. Provides a basic scene management and a
small set of control variables (camera, LOD, style, clip) that are
used by the scene classes. Renering wrappers are available but
minimal.

There is no concept of GL-context here ... we just draw
into whatever is set from outside.

Development notes:

Each viewer automatically creates a TGLRnrCtx and passes it down
all render functions.
*/

ClassImp(TGLViewerBase);

////////////////////////////////////////////////////////////////////////////////

TGLViewerBase::TGLViewerBase() :
   fRnrCtx    (0),
   fCamera    (0),
   fClip      (0),
   fLOD       (TGLRnrCtx::kLODHigh),
   fStyle     (TGLRnrCtx::kFill),
   fWFLineW   (1),
   fOLLineW   (1),

   fResetSceneInfosOnRender (kFALSE),
   fChanged                 (kFALSE)
{
   // Constructor.

   fRnrCtx = new TGLRnrCtx(this);

   fWFLineW = gEnv->GetValue("OpenGL.WireframeLineScalingFactor", 1.0);
   fOLLineW = gEnv->GetValue("OpenGL.OutlineLineScalingFactor", 1.0);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLViewerBase::~TGLViewerBase()
{
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      (*i)->GetScene()->RemoveViewer(this);
      delete *i;
   }

   DeleteOverlayElements(TGLOverlayElement::kAll);

   delete fRnrCtx;
}

////////////////////////////////////////////////////////////////////////////////
/// Name to print in locking output.

const char* TGLViewerBase::LockIdStr() const
{
   return "TGLViewerBase";
}

/**************************************************************************/
// Scene & scene-info management
/**************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Find scene-info corresponding to scene.

TGLViewerBase::SceneInfoList_i
TGLViewerBase::FindScene(TGLSceneBase* scene)
{
   SceneInfoList_i i = fScenes.begin();
   while (i != fScenes.end() && (*i)->GetScene() != scene) ++i;
   return i;
}

////////////////////////////////////////////////////////////////////////////////
/// Add new scene, appropriate scene-info is created.

TGLSceneInfo* TGLViewerBase::AddScene(TGLSceneBase* scene)
{
   SceneInfoList_i i = FindScene(scene);
   if (i == fScenes.end()) {
      TGLSceneInfo* sinfo = scene->CreateSceneInfo(this);
      fScenes.push_back(sinfo);
      scene->AddViewer(this);
      Changed();
      return sinfo;
   } else {
      Warning("TGLViewerBase::AddScene", "scene '%s' already in the list.",
              scene->GetName());
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove scene from the viewer, its scene-info is deleted.

void TGLViewerBase::RemoveScene(TGLSceneBase* scene)
{
   SceneInfoList_i i = FindScene(scene);
   if (i != fScenes.end()) {
      delete *i;
      fScenes.erase(i);
      scene->RemoveViewer(this);
      Changed();
   } else {
      Warning("TGLViewerBase::RemoveScene", "scene '%s' not found.",
              scene->GetName());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all scenes from the viewer, their scene-infos are deleted.

void TGLViewerBase::RemoveAllScenes()
{
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      TGLSceneInfo * sinfo = *i;
      sinfo->GetScene()->RemoveViewer(this);
      delete sinfo;
   }
   fScenes.clear();
   Changed();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove scene, its scene-info is deleted.
/// Called from scene that is being destroyed while still holding
/// viewer references.

void TGLViewerBase::SceneDestructing(TGLSceneBase* scene)
{
   SceneInfoList_i i = FindScene(scene);
   if (i != fScenes.end()) {
      delete *i;
      fScenes.erase(i);
      Changed();
   } else {
      Warning("TGLViewerBase::SceneDestructing", "scene not found.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find scene-info corresponding to scene.

TGLSceneInfo* TGLViewerBase::GetSceneInfo(TGLSceneBase* scene)
{
   SceneInfoList_i i = FindScene(scene);
   if (i != fScenes.end())
      return *i;
   else
      return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find logical-shape representing object id in the list of scenes.
/// Return 0 if not found.

TGLLogicalShape* TGLViewerBase::FindLogicalInScenes(TObject* id)
{
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      TGLLogicalShape *lshp = (*i)->GetScene()->FindLogical(id);
      if (lshp)
         return lshp;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add overlay element.

void TGLViewerBase::AddOverlayElement(TGLOverlayElement* el)
{
   fOverlay.push_back(el);
   Changed();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove overlay element.

void TGLViewerBase::RemoveOverlayElement(TGLOverlayElement* el)
{
   OverlayElmVec_i it = std::find(fOverlay.begin(), fOverlay.end(), el);
   if (it != fOverlay.end())
      fOverlay.erase(it);
   Changed();
}

////////////////////////////////////////////////////////////////////////////////
/// Delete overlay elements that are annotations.

void TGLViewerBase::DeleteOverlayAnnotations()
{
   DeleteOverlayElements(TGLOverlayElement::kAnnotation);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete overlay elements.

void TGLViewerBase::DeleteOverlayElements(TGLOverlayElement::ERole role)
{
   OverlayElmVec_t ovl;
   fOverlay.swap(ovl);

   for (OverlayElmVec_i i = ovl.begin(); i != ovl.end(); ++i)
   {
      if (role == TGLOverlayElement::kAll || (*i)->GetRole() == role)
         delete *i;
      else
         fOverlay.push_back(*i);
   }

   Changed();
}

/**************************************************************************/
// SceneInfo update / check
/**************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Force rebuild of view-dependent scene-info structures.
///
/// This should be called before calling render (draw/select) if
/// something that affects camera interest has been changed.

void TGLViewerBase::ResetSceneInfos()
{
   SceneInfoList_i i = fScenes.begin();
   while (i != fScenes.end())
   {
      (*i)->ResetSceneStamp();
      ++i;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Merge bounding-boxes of all active registered scenes.

void TGLViewerBase::MergeSceneBBoxes(TGLBoundingBox& bbox)
{
   bbox.SetEmpty();
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      TGLSceneInfo * sinfo = *i;
      if (sinfo->GetActive())
      {
         sinfo->SetupTransformsAndBBox(); // !!! transform not done yet, no camera
         bbox.MergeAligned(sinfo->GetTransformedBBox());
      }
   }
}

/**************************************************************************/
// Rendering / selection virtuals
/**************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Setup clip-object. Protected virtual method.

void TGLViewerBase::SetupClipObject()
{
   if (fClip)
   {
      fClip->Setup(fOverallBoundingBox);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize render-context, setup camera, GL, render-area.
/// Check and lock scenes, determine their visibility.

void TGLViewerBase::PreRender()
{
   TGLContextIdentity* cid = TGLContextIdentity::GetCurrent();
   if (cid == 0)
   {
      // Assume derived class set it up for us.
      // This happens due to complex implementation
      // of gl-in-pad using gGLManager.
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

   fRnrCtx->SetCamera        (fCamera);
   fRnrCtx->SetViewerLOD     (fLOD);
   fRnrCtx->SetViewerStyle   (fStyle);
   fRnrCtx->SetViewerWFLineW (fWFLineW);
   fRnrCtx->SetViewerOLLineW (fOLLineW);
   fRnrCtx->SetViewerClip    (fClip);

   if (fResetSceneInfosOnRender)
   {
      ResetSceneInfos();
      fResetSceneInfosOnRender = kFALSE;
   }

   fOverallBoundingBox.SetEmpty();
   SceneInfoList_t locked_scenes;
   for (SceneInfoList_i i=fScenes.begin(); i!=fScenes.end(); ++i)
   {
      TGLSceneInfo *sinfo = *i;
      TGLSceneBase *scene = sinfo->GetScene();
      if (sinfo->GetActive())
      {
         if ( ! fRnrCtx->Selection() || scene->GetSelectable())
         {
            if ( ! sinfo->GetScene()->TakeLock(kDrawLock))
            {
               Warning("TGLViewerBase::PreRender", "locking of scene '%s' failed, skipping.",
                       sinfo->GetScene()->GetName());
               continue;
            }
            locked_scenes.push_back(sinfo);
         }
         sinfo->SetupTransformsAndBBox(); // !!! transform not done yet
         fOverallBoundingBox.MergeAligned(sinfo->GetTransformedBBox());
      }
   }

   fCamera->Apply(fOverallBoundingBox, fRnrCtx->GetPickRectangle());
   SetupClipObject();

   // Make precursory selection of visible scenes.
   // Only scene bounding-box .vs. camera frustum check performed.
   fVisScenes.clear();
   for (SceneInfoList_i i=locked_scenes.begin(); i!=locked_scenes.end(); ++i)
   {
      TGLSceneInfo         * sinfo = *i;
      const TGLBoundingBox & bbox  = sinfo->GetTransformedBBox();
      Bool_t visp = (!bbox.IsEmpty() && fCamera->FrustumOverlap(bbox) != Rgl::kOutside);
      sinfo->ViewCheck(visp);
      if (visp) {
         fRnrCtx->SetSceneInfo(sinfo);
         sinfo->GetScene()->PreDraw(*fRnrCtx);
         if (sinfo->IsVisible()) {
            fVisScenes.push_back(sinfo);
         } else {
            sinfo->GetScene()->PostDraw(*fRnrCtx);
            sinfo->GetScene()->ReleaseLock(kDrawLock);
         }
         fRnrCtx->SetSceneInfo(0);
      } else {
         sinfo->GetScene()->ReleaseLock(kDrawLock);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Call sub-rendering function render_foo on all currently visible
/// scenes.

void TGLViewerBase::SubRenderScenes(SubRender_foo render_foo)
{
   Int_t nScenes = fVisScenes.size();

   for (Int_t i = 0; i < nScenes; ++i)
   {
      TGLSceneInfo* sinfo = fVisScenes[i];
      TGLSceneBase* scene = sinfo->GetScene();
      fRnrCtx->SetSceneInfo(sinfo);
      glPushName(i);
      scene->PreRender(*fRnrCtx);
      (scene->*render_foo)(*fRnrCtx);
      scene->PostRender(*fRnrCtx);
      glPopName();
      fRnrCtx->SetSceneInfo(0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Render all scenes. This is done in two main passes:
/// - render opaque objects from all scenes
/// - render transparent objects from all scenes

void TGLViewerBase::Render()
{
   RenderOpaque();
   RenderTransparent();
}

////////////////////////////////////////////////////////////////////////////////
/// Render non-selected objects from all scenes.

void TGLViewerBase::RenderNonSelected()
{
   SubRenderScenes(&TGLSceneBase::RenderOpaque);

   TGLCapabilityEnabler blend(GL_BLEND, kTRUE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDepthMask(GL_FALSE);

   SubRenderScenes(&TGLSceneBase::RenderTransp);

   glDepthMask(GL_TRUE);

   TGLUtil::CheckError("TGLViewerBase::RenderNonSelected - pre exit check");
}

////////////////////////////////////////////////////////////////////////////////
/// Render selected objects from all scenes.

void TGLViewerBase::RenderSelected()
{
   SubRenderScenes(&TGLSceneBase::RenderSelOpaque);

   TGLCapabilityEnabler blend(GL_BLEND, kTRUE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDepthMask(GL_FALSE);

   SubRenderScenes(&TGLSceneBase::RenderSelTransp);

   glDepthMask(GL_TRUE);

   TGLUtil::CheckError("TGLViewerBase::RenderSelected - pre exit check");
}

////////////////////////////////////////////////////////////////////////////////
/// Render selected objects from all scenes for highlight.

void TGLViewerBase::RenderSelectedForHighlight()
{
   fRnrCtx->SetHighlight(kTRUE);

   SubRenderScenes(&TGLSceneBase::RenderSelOpaqueForHighlight);

   TGLCapabilityEnabler blend(GL_BLEND, kTRUE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDepthMask(GL_FALSE);

   SubRenderScenes(&TGLSceneBase::RenderSelTranspForHighlight);

   glDepthMask(GL_TRUE);

   fRnrCtx->SetHighlight(kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Render opaque objects from all scenes.

void TGLViewerBase::RenderOpaque(Bool_t rnr_non_selected, Bool_t rnr_selected)
{
   if (rnr_non_selected)
   {
      SubRenderScenes(&TGLSceneBase::RenderOpaque);
   }
   if (rnr_selected)
   {
      SubRenderScenes(&TGLSceneBase::RenderSelOpaque);
   }

   TGLUtil::CheckError("TGLViewerBase::RenderOpaque - pre exit check");
}

////////////////////////////////////////////////////////////////////////////////
/// Render transparent objects from all scenes.

void TGLViewerBase::RenderTransparent(Bool_t rnr_non_selected, Bool_t rnr_selected)
{
   TGLCapabilityEnabler blend(GL_BLEND, kTRUE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDepthMask(GL_FALSE);

   if (rnr_non_selected)
   {
      SubRenderScenes(&TGLSceneBase::RenderTransp);
   }
   if (rnr_selected)
   {
      SubRenderScenes(&TGLSceneBase::RenderSelTransp);
   }

   glDepthMask(GL_TRUE);

   TGLUtil::CheckError("TGLViewerBase::RenderTransparent - pre exit check");
}

////////////////////////////////////////////////////////////////////////////////
/// Render overlay objects.

void TGLViewerBase::RenderOverlay(Int_t state, Bool_t selection)
{
   Int_t nOvl = fOverlay.size();
   for (Int_t i = 0; i < nOvl; ++i)
   {
      TGLOverlayElement* el = fOverlay[i];
      if (el->GetState() & state)
      {
         if (selection) glPushName(i);
         el->Render(*fRnrCtx);
         if (selection) glPopName();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Function called after rendering is finished.
/// Here we just unlock the scenes.

void TGLViewerBase::PostRender()
{
   for (SceneInfoVec_i i = fVisScenes.begin(); i != fVisScenes.end(); ++i)
   {
      TGLSceneInfo* sinfo = *i;
      fRnrCtx->SetSceneInfo(sinfo);
      sinfo->GetScene()->PostDraw(*fRnrCtx);
      fRnrCtx->SetSceneInfo(0);
      sinfo->GetScene()->ReleaseLock(kDrawLock);
   }
   fChanged = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Perform minimal initialization for overlay selection.
/// Here we assume that scene has already been drawn and that
/// camera and overall bounding box are ok.
/// Scenes are not locked.

void TGLViewerBase::PreRenderOverlaySelection()
{
   fCamera->Apply(fOverallBoundingBox, fRnrCtx->GetPickRectangle());
}

////////////////////////////////////////////////////////////////////////////////
/// Perform cleanup after overlay selection.

void TGLViewerBase::PostRenderOverlaySelection()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Process selection record on buffer-position 'recIdx' and
/// fill the data into 'rec'.
///
/// Returns TRUE if scene was demangled and an object identified.
/// When FALSE is returned it is still possible that scene has been
/// identified. Check for this if interested in scene-selection.
///
/// The select-buffer is taken form fRnrCtx.

Bool_t TGLViewerBase::ResolveSelectRecord(TGLSelectRecord& rec, Int_t recIdx)
{
   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();
   if (recIdx >= sb->GetNRecords())
       return kFALSE;

   if (sb->SelectRecord(rec, recIdx) < 1)
      return kFALSE;

   UInt_t sceneIdx = rec.GetItem(0);
   if (sceneIdx >= fVisScenes.size())
       return kFALSE;

   TGLSceneInfo* sinfo = fVisScenes[sceneIdx];
   rec.SetSceneInfo(sinfo);
   return sinfo->GetScene()->ResolveSelectRecord(rec, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Find next select record that can be resolved, starting from
/// position 'recIdx'.
/// 'recIdx' is passed as reference and points to found record in the buffer.

Bool_t TGLViewerBase::FindClosestRecord(TGLSelectRecord& rec, Int_t& recIdx)
{
   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();

   while (recIdx < sb->GetNRecords())
   {
      if (ResolveSelectRecord(rec, recIdx))
         return kTRUE;
      ++recIdx;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Find next select record that can be resolved and whose result is
/// not transparent, starting from position 'recIdx'.
/// 'recIdx' is passed as reference and points to found record in the buffer.

Bool_t TGLViewerBase::FindClosestOpaqueRecord(TGLSelectRecord& rec, Int_t& recIdx)
{
   TGLSelectBuffer* sb = fRnrCtx->GetSelectBuffer();

   while (recIdx < sb->GetNRecords())
   {
      if (ResolveSelectRecord(rec, recIdx) && ! rec.GetTransparent())
         return kTRUE;
      ++recIdx;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Find next overlay-select record that can be resolved, starting from
/// position 'recIdx'.
/// 'recIdx' is passed as reference and points to found record in the buffer.

Bool_t TGLViewerBase::FindClosestOverlayRecord(TGLOvlSelectRecord& rec,
                                               Int_t             & recIdx)
{
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
