// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSceneBase.h"
#include "TGLSceneInfo.h"
#include "TGLViewerBase.h"
#include "TGLRnrCtx.h"
#include "TGLCamera.h"
#include "TGLClip.h"
#include "TGLIncludes.h"

#include <TMath.h>

#include <string>
#include <algorithm>

//==============================================================================
// TGLSceneBase
//==============================================================================

//______________________________________________________________________
//
// Scene base-class --  provides basic interface expected by the
// TGLViewer or its sub-classes:
// * unique scene id
// * scene locking
// * overall bounding box
// * list of viewers displaying the scene (for update propagation)
// * virtual interface for draw/select/render (?)
//
// The standard ROOT OpenGL scene is implemented in direct sub-class
// TGLScene.
//
// Note that while each scene can be shared among several viewers, ALL
// of them are obliged to share the same display-list space (this can
// be achieved on GL-context creation time; Matevz believes that by
// default all GL contexts must use shared display-lists etc).


ClassImp(TGLSceneBase);

UInt_t TGLSceneBase::fgSceneIDSrc = 1;

//______________________________________________________________________________
TGLSceneBase::TGLSceneBase() :
   TGLLockable(),

   fTimeStamp        (1),
   fMinorStamp       (1),
   fLOD              (TGLRnrCtx::kLODHigh),
   fStyle            (TGLRnrCtx::kStyleUndef),
   fWFLineW          (0),
   fOLLineW          (0),
   fClip             (0),
   fSelectable       (kTRUE),
   fBoundingBox      (),
   fBoundingBoxValid (kFALSE),
   fDoFrustumCheck   (kTRUE),
   fDoClipCheck      (kTRUE),
   fAutoDestruct     (kTRUE)
{
   // Default constructor.

   fSceneID = fgSceneIDSrc++;
   fName = Form("unnamed-%d", fSceneID);
}

//______________________________________________________________________________
TGLSceneBase::~TGLSceneBase()
{
   // Destructor.

   for (ViewerList_i i=fViewers.begin(); i!=fViewers.end(); ++i)
   {
      (*i)->SceneDestructing(this);
   }
}

//______________________________________________________________________________
void TGLSceneBase::AddViewer(TGLViewerBase* viewer)
{
   // Add viewer to the list.

   ViewerList_i i = std::find(fViewers.begin(), fViewers.end(), viewer);
   if (i == fViewers.end())
      fViewers.push_back(viewer);
   else
      Warning("TGLSceneBase::AddViewer", "viewer already in the list.");
}

//______________________________________________________________________________
void TGLSceneBase::RemoveViewer(TGLViewerBase* viewer)
{
   // Remove viewer from the list.
   // If auto-destruct is on and the last viewer is removed the scene
   // destructs itself.

   ViewerList_i i = std::find(fViewers.begin(), fViewers.end(), viewer);
   if (i != fViewers.end())
      fViewers.erase(i);
   else
      Warning("TGLSceneBase::RemoveViewer", "viewer not found in the list.");

   if (fViewers.empty() && fAutoDestruct)
   {
      if (gDebug > 0)
         Info("TGLSceneBase::RemoveViewer", "scene '%s' not used - autodestructing.", GetName());
      delete this;
   }
}
//______________________________________________________________________________
void TGLSceneBase::TagViewersChanged()
{
   // Tag all viewers as changed.

   for (ViewerList_i i=fViewers.begin(); i!=fViewers.end(); ++i)
   {
      (*i)->Changed();
   }
}

/**************************************************************************/

//______________________________________________________________________________
const char* TGLSceneBase::LockIdStr() const
{
   // Name printed on locking info messages.

   return Form("TGLSceneBase %s", fName.Data());
}

/**************************************************************************/
// SceneInfo management
/**************************************************************************/

//______________________________________________________________________________
TGLSceneInfo* TGLSceneBase::CreateSceneInfo(TGLViewerBase* view)
{
   // Create a scene-info instance appropriate for this scene class.
   // Here we instantiate the scene-info base-class TGLSceneInfo.

   return new TGLSceneInfo(view, this);
}

//______________________________________________________________________________
void TGLSceneBase::RebuildSceneInfo(TGLRnrCtx& ctx)
{
   // Fill scene-info with very basic information that is practically
   // view independent. This is called when scene content is changed
   // or when camera-interest changes.

   TGLSceneInfo* sinfo = ctx.GetSceneInfo();

   sinfo->SetLastClip(0);
   sinfo->SetLastCamera(0);
}

//______________________________________________________________________________
void TGLSceneBase::UpdateSceneInfo(TGLRnrCtx& ctx)
{
   // Fill scene-info with information needed for rendering, take into
   // account the render-context (viewer state, camera, clipping).
   // Usually called from TGLViewer before rendering a scene if some
   // moderately significant part of render-context has changed.
   //
   // Here we update the basic state (clear last-LOD, mark the time,
   // set global <-> scene transforamtion matrices) and potentially
   // study and refine the clipping planes based on scene bounding box.

   if (gDebug > 3)
   {
      Info("TGLSceneBase::UpdateSceneInfo",
           "'%s' timestamp=%u",
           GetName(), fTimeStamp);
   }

   TGLSceneInfo* sinfo = ctx.GetSceneInfo();

   // ------------------------------------------------------------
   // Reset
   // ------------------------------------------------------------

   sinfo->SetLastLOD   (TGLRnrCtx::kLODUndef);
   sinfo->SetLastStyle (TGLRnrCtx::kStyleUndef);
   sinfo->SetSceneStamp(fTimeStamp);

   sinfo->InFrustum (kTRUE);
   sinfo->InClip    (kTRUE);
   sinfo->ClipMode  (TGLSceneInfo::kClipNone);

   // ------------------------------------------------------------
   // Setup
   // ------------------------------------------------------------

   // !!!
   // setup scene transformation matrices
   // so far the matrices in scene-base and scene-info are not enabled
   // sinfo->fSceneToGlobal = scene-info-trans * scene-base-trans;
   // sinfo->fGlobalToScene = inv of above;
   // transform to clip and to eye coordinates also interesting
   //
   // All these are now done in TGLViewerBase::PreRender() via
   // TGLSceneInfo::SetupTransformsAndBBox().

   sinfo->SetLastClip(0);
   sinfo->FrustumPlanes().clear();
   sinfo->ClipPlanes().clear();

   if (fDoFrustumCheck)
   {
      for (Int_t i=0; i<TGLCamera::kPlanesPerFrustum; ++i)
      {
         TGLPlane p = ctx.GetCamera()->FrustumPlane((TGLCamera::EFrustumPlane)i);
         // !!! transform plane
         switch (BoundingBox().Overlap(p))
         {
            case kInside:  // Whole scene passes ... no need to store it.
               break;
            case kPartial:
               sinfo->FrustumPlanes().push_back(p);
               break;
            case kOutside:
               sinfo->InFrustum(kFALSE);
               break;
         }
      }
   }

   if (fDoClipCheck && ctx.HasClip())
   {
      if (ctx.Clip()->GetMode() == TGLClip::kOutside)
         sinfo->ClipMode(TGLSceneInfo::kClipOutside);
      else
         sinfo->ClipMode(TGLSceneInfo::kClipInside);

      std::vector<TGLPlane> planeSet;
      ctx.Clip()->PlaneSet(planeSet);

      // Strip any planes outside the scene bounding box - no effect
      std::vector<TGLPlane>::iterator it = planeSet.begin();
      while (it != planeSet.end())
      {
         // !!! transform plane
         switch (BoundingBox().Overlap(*it))
         {
            case kInside:  // Whole scene passes ... no need to store it.
               break;
            case kPartial:
               sinfo->ClipPlanes().push_back(*it);
               break;
            case kOutside: // Depends on mode
               if (sinfo->ClipMode() == TGLSceneInfo::kClipOutside)
               {
                  // Scene is outside of whole clip object - nothing visible.
                  sinfo->InClip(kFALSE);
               }
               else
               {
                  // Scene is completely inside of whole clip object -
                  // draw all scene without clipping.
                  sinfo->ClipMode(TGLSceneInfo::kClipNone);
               }
               // In either case further checks not needed.
               sinfo->ClipPlanes().clear();
               return;
         }
         ++it;
      }
      sinfo->SetLastClip(ctx.Clip());
      sinfo->SetClipStamp(ctx.Clip()->TimeStamp());
   }

   sinfo->SetLastCamera(ctx.GetCamera());
   sinfo->SetCameraStamp(ctx.GetCamera()->TimeStamp());
}

//______________________________________________________________________________
void TGLSceneBase::LodifySceneInfo(TGLRnrCtx& ctx)
{
   // Setup LOD-dependant values in scene-info.
   //
   // Nothing to be done here but to store the last LOD.

   if (gDebug > 3)
   {
      Info("TGLSceneBase::LodifySceneInfo",
           "'%s' timestamp=%u lod=%d",
           GetName(), fTimeStamp, ctx.CombiLOD());
   }

   TGLSceneInfo & sInfo = * ctx.GetSceneInfo();
   sInfo.SetLastLOD(ctx.CombiLOD());
}


/**************************************************************************/
// Rendering
/**************************************************************************/

//______________________________________________________________________________
void TGLSceneBase::PreDraw(TGLRnrCtx & rnrCtx)
{
   // Perform basic pre-render initialization:
   //  - calculate LOD, Style, Clipping,
   //  - build draw lists.
   //
   // This is called in the beginning of the GL-viewer draw cycle.

   if ( ! IsDrawOrSelectLock()) {
      Error("TGLSceneBase::FullRender", "expected Draw or Select Lock");
   }

   TGLSceneInfo& sInfo = * rnrCtx.GetSceneInfo();

   // Bounding-box check done elsewhere (in viewer::pre-render)

   if (fTimeStamp > sInfo.SceneStamp())
   {
      RebuildSceneInfo(rnrCtx);
   }


   Bool_t needUpdate =  sInfo.HasUpdateTimeouted();

   if (rnrCtx.GetCamera() != sInfo.LastCamera())
   {
      sInfo.ResetCameraStamp();
      needUpdate = kTRUE;
   }
   else if (rnrCtx.GetCamera()->TimeStamp() > sInfo.CameraStamp())
   {
      needUpdate = kTRUE;
   }

   TGLClip* clip = 0;
   if (sInfo.Clip() != 0) clip = sInfo.Clip();
   else if (fClip   != 0) clip = fClip;
   else                   clip = rnrCtx.ViewerClip();
   if (clip != sInfo.LastClip())
   {
      sInfo.ResetClipStamp();
      needUpdate = kTRUE;
   }
   else if (clip && clip->TimeStamp() > sInfo.ClipStamp())
   {
      needUpdate = kTRUE;
   }
   rnrCtx.SetClip(clip);

   if (needUpdate)
   {
      UpdateSceneInfo(rnrCtx);
   }


   // Setup LOD ... optionally lodify.
   Short_t lod;
   if (sInfo.LOD() != TGLRnrCtx::kLODUndef) lod = sInfo.LOD();
   else if  (fLOD  != TGLRnrCtx::kLODUndef) lod = fLOD;
   else                                     lod = rnrCtx.ViewerLOD();
   rnrCtx.SetSceneLOD(lod);
   rnrCtx.SetCombiLOD(TMath::Min(rnrCtx.ViewerLOD(), rnrCtx.SceneLOD()));
   if (needUpdate || rnrCtx.CombiLOD() != sInfo.LastLOD())
   {
      LodifySceneInfo(rnrCtx);
   }

   // Setup style.
   Short_t style;
   if (sInfo.Style() != TGLRnrCtx::kStyleUndef) style = sInfo.Style();
   else if  (fStyle  != TGLRnrCtx::kStyleUndef) style = fStyle;
   else                                         style = rnrCtx.ViewerStyle();
   rnrCtx.SetSceneStyle(style);
   sInfo.SetLastStyle(style);

   // Wireframe line width.
   Float_t wf_linew;
   if (sInfo.WFLineW() != 0) wf_linew = sInfo.WFLineW();
   else if  (fWFLineW  != 0) wf_linew = fWFLineW;
   else                      wf_linew = rnrCtx.ViewerWFLineW();
   rnrCtx.SetSceneWFLineW(wf_linew);
   sInfo.SetLastWFLineW(wf_linew);
   // Outline line width.
   Float_t ol_linew;
   if (sInfo.OLLineW() != 0) ol_linew = sInfo.OLLineW();
   else if  (fOLLineW  != 0) ol_linew = fOLLineW;
   else                      ol_linew = rnrCtx.ViewerOLLineW();
   rnrCtx.SetSceneOLLineW(ol_linew);
   sInfo.SetLastOLLineW(ol_linew);
}

//______________________________________________________________________________
void TGLSceneBase::PreRender(TGLRnrCtx & rnrCtx)
{
   // Perform pre-render initialization - fill rnrCtx with
   // values stored during PreDraw().
   //
   // This is called each time before RenderXyzz().

   TGLSceneInfo& sInfo = * rnrCtx.GetSceneInfo();

   rnrCtx.SetClip         (sInfo.LastClip());
   rnrCtx.SetCombiLOD     (sInfo.LastLOD());
   rnrCtx.SetSceneStyle   (sInfo.LastStyle());
   rnrCtx.SetSceneWFLineW (sInfo.LastWFLineW());
   rnrCtx.SetSceneOLLineW (sInfo.LastOLLineW());

   // !!!
   // eventually handle matrix stack.
   // glPushMatrix();
   // glMultMatrix(something-from-scene-info);
   // Should also fix camera matrices
}

//______________________________________________________________________________
void TGLSceneBase::Render(TGLRnrCtx & rnrCtx)
{
   // This function does rendering of all stages, the shapes are
   // rendered in the following order: opaque, transparent,
   // selected-opaque, selected-transparent.
   //
   // GL-depth buffer is cleared after transparent shapes have been
   // rendered.
   //
   // This is never called from ROOT GL directly. Use it if you know
   // you are rendering a single scene.

   RenderOpaque(rnrCtx);
   RenderTransp(rnrCtx);
   RenderSelOpaque(rnrCtx);
   RenderSelTransp(rnrCtx);
}

//______________________________________________________________________________
void TGLSceneBase::RenderOpaque(TGLRnrCtx & /*rnrCtx*/)
{
   // Render opaque elements.
}

//______________________________________________________________________________
void TGLSceneBase::RenderTransp(TGLRnrCtx & /*rnrCtx*/)
{
   // Render transparent elements.
}

//______________________________________________________________________________
void TGLSceneBase::RenderSelOpaque(TGLRnrCtx & /*rnrCtx*/)
{
   // Render selected opaque elements.
}

//______________________________________________________________________________
void TGLSceneBase::RenderSelTransp(TGLRnrCtx & /*rnrCtx*/)
{
   // Render selected transparent elements.
}

//______________________________________________________________________________
void TGLSceneBase::PostRender(TGLRnrCtx & /*rnrCtx*/)
{
   // Perform post-render clean-up.

   // !!!
   // Cleanup matrix stack
   // glPopMatrix();
   // Should also fix camera matrices
}

//______________________________________________________________________________
void TGLSceneBase::PostDraw(TGLRnrCtx & /*rnrCtx*/)
{
   // Finalize drawing.
   //
   // This is called at the end of the GL-viewer draw cycle.
}

/**************************************************************************/
// Selection
/**************************************************************************/

//______________________________________________________________________________
Bool_t TGLSceneBase::ResolveSelectRecord(TGLSelectRecord & /*rec*/,
                                         Int_t             /*curIdx*/)
{
   // Process selection record rec.
   // 'curIdx' is the item position where the scene should start
   // its processing.
   // Return TRUE if an object has been identified or FALSE otherwise.
   // The scene-info member of the record is already set by the caller.
   //
   // See implementation in sub-class TGLScene, here we just return FALSE.

   return kFALSE;
}
