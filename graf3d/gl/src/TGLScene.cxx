// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007
// Author:  Richard Maunder  25/05/2005
// Parts taken from original TGLRender by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLScene.h"
#include "TGLRnrCtx.h"
#include "TGLObject.h"
#include "TGLSelectRecord.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLCamera.h"
#include "TGLContext.h"
#include "TGLIncludes.h"

#include <TColor.h>
#include <TROOT.h>
#include <TClass.h>

#include <algorithm>

//==============================================================================
// TGLScene::TSceneInfo
//==============================================================================

//______________________________________________________________________
//
// Extend TGLSceneInfo for needs of TGLScene:
//
// 1. DrawElement vectors for opaque/transparent shapes which cache
// physicals that pass the clip tests (frustum and additional
// clip-object);
//
// 2. Statistics / debug information
//

//______________________________________________________________________________
TGLScene::TSceneInfo::TSceneInfo(TGLViewerBase* view, TGLScene* scene) :
   TGLSceneInfo (view, scene),
   fMinorStamp  (0),
   fOpaqueCnt   (0),
   fTranspCnt   (0),
   fAsPixelCnt  (0)
{
   // Constructor.
}

//______________________________________________________________________________
TGLScene::TSceneInfo::~TSceneInfo()
{
   // Destructor.
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::ClearDrawElementVec(DrawElementVec_t& vec,
                                               Int_t maxSize)
{
   // Clear given vec and if it grew too large compared to the size of
   // shape-of-interest also resize it.

   if (vec.capacity() > (size_t) maxSize) {
      DrawElementVec_t foo;
      foo.reserve((size_t) maxSize);
      vec.swap(foo);
   } else {
      vec.clear();
   }
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::ClearDrawElementPtrVec(DrawElementPtrVec_t& vec,
                                                  Int_t maxSize)
{
   // Clear given vec and if it grew too large compared to the size of
   // shape-of-interest also resize it.

   if (vec.capacity() > (size_t) maxSize) {
      DrawElementPtrVec_t foo;
      foo.reserve((size_t) maxSize);
      vec.swap(foo);
   } else {
      vec.clear();
   }
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::ClearAfterRebuild()
{
   // Clear DrawElementVector fVisibleElement and optionally resize it
   // so that it doesn't take more space then required by all the
   // elements in the scene's draw-list.

   Int_t maxSize = (Int_t) fShapesOfInterest.size();

   ClearDrawElementVec(fVisibleElements, maxSize);
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::ClearAfterUpdate()
{
   // Clear DrawElementPtrVectors and optionally resize them so that
   // they don't take more space then required by all the elements in
   // the scene's draw-list.

   Int_t maxSize = (Int_t) fShapesOfInterest.size();

   ClearDrawElementPtrVec(fOpaqueElements, maxSize);
   ClearDrawElementPtrVec(fTranspElements, maxSize);
   ClearDrawElementPtrVec(fSelOpaqueElements, maxSize);
   ClearDrawElementPtrVec(fSelTranspElements, maxSize);

   fMinorStamp = 0;
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::Lodify(TGLRnrCtx& ctx)
{
   // Quantize LODs for gice render-context.

   for (DrawElementVec_i i = fVisibleElements.begin(); i != fVisibleElements.end(); ++i)
      i->fPhysical->QuantizeShapeLOD(i->fPixelLOD, ctx.CombiLOD(), i->fFinalLOD);
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::PreDraw()
{
   // Prepare for drawing - fill DrawElementPtrVectors from the
   // contents of fVisibleElements if there was some change.

   if (fMinorStamp < fScene->GetMinorStamp())
   {
      fOpaqueElements.clear();
      fTranspElements.clear();
      fSelOpaqueElements.clear();
      fSelTranspElements.clear();

      for (DrawElementVec_i i = fVisibleElements.begin(); i != fVisibleElements.end(); ++i)
      {
         if (i->fPhysical->IsSelected())
         {
            if (i->fPhysical->IsTransparent())
               fSelTranspElements.push_back(&*i);
            else
               fSelOpaqueElements.push_back(&*i);
         } else {
            if (i->fPhysical->IsTransparent())
               fTranspElements.push_back(&*i);
            else
               fOpaqueElements.push_back(&*i);
         }
      }
      fMinorStamp = fScene->GetMinorStamp();
   }
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::PostDraw()
{
   // Clean-up after drawing, nothing to be done here.
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::ResetDrawStats()
{
   // Reset draw statistics.

   fOpaqueCnt  = 0;
   fTranspCnt  = 0;
   fAsPixelCnt = 0;
   fByShapeCnt.clear();
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::UpdateDrawStats(const TGLPhysicalShape& shape,
                                           Short_t lod)
{
   // Update draw stats, for newly drawn 'shape'

   // Update opaque/transparent draw count
   if (shape.IsTransparent()) {
      ++fTranspCnt;
   } else {
      ++fOpaqueCnt;
   }

   if (lod == TGLRnrCtx::kLODPixel) {
      ++fAsPixelCnt;
   }

   // By type only done at higher debug level.
   if (gDebug>3) {
      // Update the stats
      TClass* logIsA = shape.GetLogical()->IsA();
      std::map<TClass*, UInt_t>::iterator it = fByShapeCnt.find(logIsA);
      if (it == fByShapeCnt.end()) {
         //do not need to check insert(.....).second, because it was stats.end() before
         it = fByShapeCnt.insert(std::make_pair(logIsA, 0u)).first;
      }

      it->second++;
   }
}

//______________________________________________________________________________
void TGLScene::TSceneInfo::DumpDrawStats()
{
   // Output draw stats to Info stream.

   if (gDebug>2)
   {
      TString out;
      // Draw/container counts
      out += Form("Drew scene (%s / %i LOD) - %i (Op %i Trans %i) %i pixel\n",
                  TGLRnrCtx::StyleName(LastStyle()), LastLOD(),
                  fOpaqueCnt + fTranspCnt, fOpaqueCnt, fTranspCnt, fAsPixelCnt);
      out += Form("\tInner phys nums: physicals=%d, of_interest=%lu, visible=%lu, op=%lu, trans=%lu",
                  ((TGLScene*)fScene)->GetMaxPhysicalID(),
                  (ULong_t)fShapesOfInterest.size(), (ULong_t)fVisibleElements.size(),
                  (ULong_t)fOpaqueElements.size(), (ULong_t)fTranspElements.size());

      // By shape type counts
      if (gDebug>3)
      {
         out += "\n\tStatistics by shape:\n";
         std::map<TClass*, UInt_t>::const_iterator it = fByShapeCnt.begin();
         while (it != fByShapeCnt.end()) {
            out += Form("\t%-20s  %u\n", it->first->GetName(), it->second);
            it++;
         }
      }
      Info("TGLScene::DumpDrawStats()", "%s",out.Data());
   }
}


//==============================================================================
// TGLScene
//==============================================================================

//______________________________________________________________________________
//
// TGLScene provides managememnt and rendering of ROOT's default 3D
// object representation as logical and physical shapes.
//
// A GL scene is the container for all the viewable objects (shapes)
// loaded into the viewer. It consists of two main stl::maps containing
// the TGLLogicalShape and TGLPhysicalShape collections, and interface
// functions enabling viewers to manage objects in these. The physical
// shapes defined the placement of copies of the logical shapes - see
// TGLLogicalShape/TGLPhysicalShape for more information on relationship
//
// The scene can be drawn by owning viewer, passing camera, draw style
// & quality (LOD), clipping etc - see Draw(). The scene can also be
// drawn for selection in similar fashion - see Select(). The scene
// keeps track of a single selected physical - which can be modified by
// viewers.
//
// The scene maintains a lazy calculated bounding box for the total
// scene extents, axis aligned round TGLPhysicalShape shapes.
//
// Currently a scene is owned exclusively by one viewer - however it is
// intended that it could easily be shared by multiple viewers - for
// efficiency and syncronisation reasons. Hence viewer variant objects
// camera, clips etc being owned by viewer and passed at draw/select

ClassImp(TGLScene);

//______________________________________________________________________________
TGLScene::TGLScene() :
   TGLSceneBase(),
   fGLCtxIdentity(0),
   fInSmartRefresh(kFALSE),
   fLastPointSizeScale (0),
   fLastLineWidthScale (0)
{}

//______________________________________________________________________________
TGLScene::~TGLScene()
{
   // Destroy scene objects
   TakeLock(kModifyLock);
   ReleaseGLCtxIdentity();
   DestroyPhysicals();
   DestroyLogicals();
   if (fGLCtxIdentity)
      fGLCtxIdentity->ReleaseClient();
   ReleaseLock(kModifyLock);
}

/**************************************************************************/
// GLCtxIdentity
/**************************************************************************/

//______________________________________________________________________________
void TGLScene::ReleaseGLCtxIdentity()
{
   // Release all GL resources for current context identity.
   // Requires iteration over all logical shapes.

   if (fGLCtxIdentity == 0) return;

   if (fGLCtxIdentity->IsValid())
   {
      // Purge logical's DLs
      LogicalShapeMapIt_t lit = fLogicalShapes.begin();
      while (lit != fLogicalShapes.end()) {
         lit->second->DLCachePurge();
         ++lit;
      }
   }
   else
   {
      // Drop logical's DLs
      LogicalShapeMapIt_t lit = fLogicalShapes.begin();
      while (lit != fLogicalShapes.end()) {
         lit->second->DLCacheDrop();
         ++lit;
      }
   }
   fGLCtxIdentity->ReleaseClient();
   fGLCtxIdentity = 0;
}

/**************************************************************************/
// SceneInfo management
/**************************************************************************/


//______________________________________________________________________________
TGLScene::TSceneInfo* TGLScene::CreateSceneInfo(TGLViewerBase* view)
{
   // Create a scene-info instance appropriate for this scene class.
   // Here we instantiate the inner class TSceneInfo that includes
   // camera/clipping specific draw-list containers.

   return new TSceneInfo(view, this);
}

//______________________________________________________________________________
inline Bool_t TGLScene::ComparePhysicalVolumes(const TGLPhysicalShape* shape1,
                                               const TGLPhysicalShape* shape2)
{
   // Compare 'shape1' and 'shape2' bounding box volumes - return kTRUE if
   // 'shape1' bigger than 'shape2'.

   return (shape1->BoundingBox().Volume() > shape2->BoundingBox().Volume());
}

//______________________________________________________________________________
inline Bool_t TGLScene::ComparePhysicalDiagonals(const TGLPhysicalShape* shape1,
                                                 const TGLPhysicalShape* shape2)
{
   // Compare 'shape1' and 'shape2' bounding box volumes - return kTRUE if
   // 'shape1' bigger than 'shape2'.

   return (shape1->BoundingBox().Diagonal() > shape2->BoundingBox().Diagonal());
}

//______________________________________________________________________________
void TGLScene::RebuildSceneInfo(TGLRnrCtx& rnrCtx)
{
   // Major change in scene, need to rebuild all-element draw-vector and
   // sort it.
   //
   // Sort the TGLPhysical draw list by shape bounding box diagonal, from
   // large to small. This makes dropout of shapes with time limited
   // Draw() calls must less noticable. As this does not use projected
   // size it only needs to be done after a scene content change - not
   // everytime scene drawn (potential camera/projection change).

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (sinfo == 0 || sinfo->GetScene() != this) {
      Error("TGLScene::RebuildSceneInfo", "Scene mismatch.");
      return;
   }

   TGLSceneBase::RebuildSceneInfo(rnrCtx);

   if (sinfo->fShapesOfInterest.capacity() > fPhysicalShapes.size()) {
      ShapeVec_t foo;
      foo.reserve(fPhysicalShapes.size());
      sinfo->fShapesOfInterest.swap(foo);
   } else {
      sinfo->fShapesOfInterest.clear();
   }

   PhysicalShapeMapIt_t pit = fPhysicalShapes.begin();
   while (pit != fPhysicalShapes.end())
   {
      TGLPhysicalShape      * pshp = pit->second;
      const TGLLogicalShape * lshp = pshp->GetLogical();
      if (rnrCtx.GetCamera()->OfInterest(pshp->BoundingBox(),
                                         lshp->IgnoreSizeForOfInterest()))
      {
         sinfo->fShapesOfInterest.push_back(pshp);
      }
      ++pit;
   }

   std::sort(sinfo->fShapesOfInterest.begin(), sinfo->fShapesOfInterest.end(),
             TGLScene::ComparePhysicalDiagonals);

   sinfo->ClearAfterRebuild();
}

//______________________________________________________________________________
void TGLScene::UpdateSceneInfo(TGLRnrCtx& rnrCtx)
{
   // Fill scene-info with information needed for rendering, take into
   // account the render-context (viewer state, camera, clipping).
   // Here we have to iterate over all the physical shapes and select
   // the visible ones. While at it, opaque and transparent shapes are
   // divided into two groups.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (sinfo == 0 || sinfo->GetScene() != this) {
      Error("TGLScene::UpdateSceneInfo", "Scene mismatch.");
      return;
   }

   // Clean-up/reset, update of transformation matrices and clipping
   // planes done in base-class.
   TGLSceneBase::UpdateSceneInfo(rnrCtx);

   if (!sinfo->IsVisible())
      return;

   sinfo->fVisibleElements.clear();

   // Check individual physicals, build DrawElementList.

   Int_t  checkCount = 0;
   Bool_t timerp     = rnrCtx.IsStopwatchRunning();
   sinfo->ResetUpdateTimeouted();

   for (ShapeVec_i phys=sinfo->fShapesOfInterest.begin();
        phys!=sinfo->fShapesOfInterest.end();
        ++phys, ++checkCount)
   {
      const TGLPhysicalShape * drawShape = *phys;

      // TODO: Do small skipping first? Probably cheaper than frustum check
      // Profile relative costs? The frustum check could be done implictly
      // from the LOD as we project all 8 verticies of the BB onto viewport

      // Work out if we need to draw this shape - assume we do first
      Bool_t drawNeeded = kTRUE;

      // Draw test against passed clipping planes.
      // Do before camera clipping on assumption clip planes remove
      // more objects.
      if (sinfo->ClipMode() == TGLSceneInfo::kClipOutside)
      {
         // Draw not needed if outside any of the planes.
         std::vector<TGLPlane>::iterator pi = sinfo->ClipPlanes().begin();
         while (pi != sinfo->ClipPlanes().end())
         {
            if (drawShape->BoundingBox().Overlap(*pi) == Rgl::kOutside)
            {
               drawNeeded = kFALSE;
               break;
            }
            ++pi;
         }
      }
      else if (sinfo->ClipMode() == TGLSceneInfo::kClipInside)
      {
         // Draw not needed if inside all the planes.
         std::vector<TGLPlane>::iterator pi = sinfo->ClipPlanes().begin();
         size_t cnt = 0;
         while (pi != sinfo->ClipPlanes().end())
         {
            Rgl::EOverlap ovlp = drawShape->BoundingBox().Overlap(*pi);
            if (ovlp == Rgl::kOutside)
               break;
            else if (ovlp == Rgl::kInside)
               ++cnt;
            ++pi;
         }
         if (cnt == sinfo->ClipPlanes().size())
            drawNeeded = kFALSE;
      }

      // Test against camera frustum planes (here mode is Outside
      // implicitly).
      if (drawNeeded)
      {
         std::vector<TGLPlane>::iterator pi = sinfo->FrustumPlanes().begin();
         while (pi != sinfo->FrustumPlanes().end())
         {
            if (drawShape->BoundingBox().Overlap(*pi) == Rgl::kOutside)
            {
               drawNeeded = kFALSE;
               break;
            }
            ++pi;
         }
      }

      // Draw? Then calculate lod and store ...
      if (drawNeeded)
      {
         DrawElement_t de(drawShape);
         drawShape->CalculateShapeLOD(rnrCtx, de.fPixelSize, de.fPixelLOD);
         sinfo->fVisibleElements.push_back(de);
      }

      // Terminate the traversal if over scene rendering limit.
      // Only test every 5000 objects as this is somewhat costly.
      if (timerp && (checkCount % 5000) == 0 && rnrCtx.HasStopwatchTimedOut())
      {
         sinfo->UpdateTimeouted();
         if (rnrCtx.ViewerLOD() == TGLRnrCtx::kLODHigh)
            Warning("TGLScene::UpdateSceneInfo",
                    "Timeout reached, not all elements processed.");
         break;
      }
   }

   sinfo->ClearAfterUpdate();

   // !!! MT Transparents should be sorted by their eye z-coordinate.
   // Need combined matrices in scene-info to do this.
   // Even more ... should z-sort contributions from ALL scenes!
}

//______________________________________________________________________________
void TGLScene::LodifySceneInfo(TGLRnrCtx& rnrCtx)
{
   // Setup LOD-dependant values in scene-info.
   // We have to perform LOD quantization for all draw-elements.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (sinfo == 0 || sinfo->GetScene() != this) {
      Error("TGLScene::LodifySceneInfo", "Scene mismatch.");
      return;
   }

   TGLSceneBase::LodifySceneInfo(rnrCtx);

   sinfo->Lodify(rnrCtx);
}


/**************************************************************************/
// Rendering
/**************************************************************************/

//______________________________________________________________________________
void TGLScene::PreDraw(TGLRnrCtx& rnrCtx)
{
   // Initialize rendering.
   // Pass to base-class where most work is done.
   // Check if GL-ctx is shared with the previous one; if not
   // wipe display-lists of all logicals.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (sinfo == 0 || sinfo->GetScene() != this) {
      TGLSceneInfo* si = rnrCtx.GetSceneInfo();
      Error("TGLScene::PreDraw", "%s", Form("SceneInfo mismatch (0x%lx, '%s').",
                                      (ULong_t)si, si ? si->IsA()->GetName() : "<>"));
      return;
   }

   // Setup ctx, check if Update/Lodify needed.
   TGLSceneBase::PreDraw(rnrCtx);

   TGLContextIdentity* cid = rnrCtx.GetGLCtxIdentity();
   if (cid != fGLCtxIdentity)
   {
      ReleaseGLCtxIdentity();
      fGLCtxIdentity = cid;
      fGLCtxIdentity->AddClientRef();
   }
   else
   {
      if (fLastPointSizeScale != TGLUtil::GetPointSizeScale() ||
          fLastLineWidthScale != TGLUtil::GetLineWidthScale())
      {
         // Clear logical's DLs
         LogicalShapeMapIt_t lit = fLogicalShapes.begin();
         while (lit != fLogicalShapes.end()) {
            lit->second->DLCacheClear();
            ++lit;
         }
      }
   }
   fLastPointSizeScale = TGLUtil::GetPointSizeScale();
   fLastLineWidthScale = TGLUtil::GetLineWidthScale();

   sinfo->PreDraw();

   // Reset-scene-info counters.
   sinfo->ResetDrawStats();
}

//______________________________________________________________________________
void TGLScene::RenderOpaque(TGLRnrCtx& rnrCtx)
{
   // Render opaque elements.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (!sinfo->fOpaqueElements.empty())
      RenderAllPasses(rnrCtx, sinfo->fOpaqueElements, kTRUE);
}

//______________________________________________________________________________
void TGLScene::RenderTransp(TGLRnrCtx& rnrCtx)
{
   // Render transparent elements.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (!sinfo->fTranspElements.empty())
      RenderAllPasses(rnrCtx, sinfo->fTranspElements, kTRUE);
}

//______________________________________________________________________________
void TGLScene::RenderSelOpaque(TGLRnrCtx& rnrCtx)
{
   // Render selected opaque elements.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if ( ! sinfo->fSelOpaqueElements.empty())
      RenderAllPasses(rnrCtx, sinfo->fSelOpaqueElements, kFALSE);
}

//______________________________________________________________________________
void TGLScene::RenderSelTransp(TGLRnrCtx& rnrCtx)
{
   // Render selected transparent elements.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (!sinfo->fSelTranspElements.empty())
      RenderAllPasses(rnrCtx, sinfo->fSelTranspElements, kFALSE);
}

//______________________________________________________________________________
void TGLScene::RenderSelOpaqueForHighlight(TGLRnrCtx& rnrCtx)
{
   // Render selected opaque elements for highlight.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if ( ! sinfo->fSelOpaqueElements.empty())
      RenderHighlight(rnrCtx, sinfo->fSelOpaqueElements);
}

//______________________________________________________________________________
void TGLScene::RenderSelTranspForHighlight(TGLRnrCtx& rnrCtx)
{
   // Render selected transparent elements for highlight.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   if (!sinfo->fSelTranspElements.empty())
      RenderHighlight(rnrCtx, sinfo->fSelTranspElements);
}

//______________________________________________________________________________
void TGLScene::RenderHighlight(TGLRnrCtx&           rnrCtx,
                               DrawElementPtrVec_t& elVec)
{
   DrawElementPtrVec_t svec(1);

   glEnable(GL_STENCIL_TEST);
   for (DrawElementPtrVec_i i = elVec.begin(); i != elVec.end(); ++i)
   {
      svec[0] = *i;

      glStencilFunc(GL_ALWAYS, 0x1, 0x1);
      glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
      glClear(GL_STENCIL_BUFFER_BIT);

      glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

      RenderAllPasses(rnrCtx, svec, kFALSE);

      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

      glStencilFunc(GL_NOTEQUAL, 0x1, 0x1);
      glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

      rnrCtx.SetHighlightOutline(kTRUE);
      RenderAllPasses(rnrCtx, svec, kFALSE);
      rnrCtx.SetHighlightOutline(kFALSE);
   }
   glDisable(GL_STENCIL_TEST);
}

//______________________________________________________________________________
void TGLScene::PostDraw(TGLRnrCtx& rnrCtx)
{
   // Called after the rendering is finished.
   // In debug mode draw statistcs is dumped.
   // Parent's PostDraw is called for GL cleanup.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());

   if (gDebug)
      sinfo->DumpDrawStats();

   sinfo->PostDraw();

   TGLSceneBase::PostDraw(rnrCtx);
}

//______________________________________________________________________________
void TGLScene::RenderAllPasses(TGLRnrCtx&           rnrCtx,
                               DrawElementPtrVec_t& elVec,
                               Bool_t               check_timeout)
{
   // Do full rendering of scene.
   //
   // First draw the opaques, then the transparents. For each we do
   // the number of passes required by draw mode and clipping setup.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   assert(sinfo != 0);

   Short_t sceneStyle = rnrCtx.SceneStyle();

   // Setup GL for current draw style - fill, wireframe, outline
   Int_t        reqPasses  = 1; // default

   Short_t      rnrPass[2];
   rnrPass[0] = rnrPass[1] = TGLRnrCtx::kPassUndef;

   switch (sceneStyle)
   {
      case TGLRnrCtx::kFill:
      case TGLRnrCtx::kOutline:
      {
         glEnable(GL_LIGHTING);
         if (sinfo->ShouldClip())
         {
            // Clip object - two sided lighting, two side polygons, don't cull (BACK) faces
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_CULL_FACE);
         }
         // No clip - default single side lighting,
         // front polygons, cull (BACK) faces ok
         if (sceneStyle == TGLRnrCtx::kOutline && ! (rnrCtx.Selection() || rnrCtx.Highlight()))
         {
            reqPasses = 2;   // Outline needs two full draws
            rnrPass[0] = TGLRnrCtx::kPassOutlineFill;
            rnrPass[1] = TGLRnrCtx::kPassOutlineLine;
         }
         else
         {
            rnrPass[0] = TGLRnrCtx::kPassFill;
         }
         break;
      }
      case TGLRnrCtx::kWireFrame:
      {
         rnrPass[0] = TGLRnrCtx::kPassWireFrame;
         glDisable(GL_LIGHTING);
         glDisable(GL_CULL_FACE);
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         break;
      }
      default:
      {
         assert(kFALSE);
      }
   }

   for (Int_t i = 0; i < reqPasses; ++i)
   {
      // For outline two full draws (fill + wireframe) required.
      // Do it this way to avoid costly GL state swaps on per drawable basis

      Short_t pass = rnrPass[i];
      rnrCtx.SetDrawPass(pass);

      if (pass == TGLRnrCtx::kPassOutlineFill)
      {
         // First pass - filled polygons
         glEnable(GL_POLYGON_OFFSET_FILL);
         glPolygonOffset(0.5f, 0.5f);
      }
      else if (pass == TGLRnrCtx::kPassOutlineLine)
      {
         // Second pass - outline (wireframe)
         TGLUtil::LineWidth(rnrCtx.SceneOLLineW());
         glDisable(GL_POLYGON_OFFSET_FILL);
         glDisable(GL_LIGHTING);

         // We are only showing back faces with clipping as a
         // better solution than completely invisible faces.
         // *Could* cull back faces and only outline on front like this:
         //    glEnable(GL_CULL_FACE);
         //    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         // However this means clipped back edges not shown - so do inside and out....
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      }
      else if (pass == TGLRnrCtx::kPassWireFrame)
      {
         TGLUtil::LineWidth(rnrCtx.SceneWFLineW());
      }

      // If no clip object no plane sets to extract/pass
      if ( ! sinfo->ShouldClip())
      {
         RenderElements(rnrCtx, elVec, check_timeout);
      }
      else
      {
         // Get the clip plane set from the clipping object
         TGLPlaneSet_t & planeSet = sinfo->ClipPlanes();

         if (gDebug > 3)
         {
            Info("TGLScene::RenderAllPasses()",
                 "%ld active clip planes", (Long_t)planeSet.size());
         }
         // Limit to smaller of plane set size or GL implementation plane support
         Int_t maxGLPlanes;
         glGetIntegerv(GL_MAX_CLIP_PLANES, &maxGLPlanes);
         UInt_t maxPlanes = maxGLPlanes;
         UInt_t planeInd;
         if (planeSet.size() < maxPlanes) {
            maxPlanes = planeSet.size();
         }

         if (sinfo->ClipMode() == TGLSceneInfo::kClipOutside)
         {
            // Clip away scene outside of the clip object.
            // Load all clip planes (up to max) at once.
            for (UInt_t ii=0; ii<maxPlanes; ii++) {
               glClipPlane(GL_CLIP_PLANE0+ii, planeSet[ii].CArr());
               glEnable(GL_CLIP_PLANE0+ii);
            }

            // Draw scene once with full time slot, physicals have been
            // clipped during UpdateSceneInfo, so no need to repeat that.
            RenderElements(rnrCtx, elVec, check_timeout);
         }
         else
         {
            // Clip away scene inside of the clip object.
            // This requires number-of-clip-planes passes and can not
            // be entirely pre-computed (non-relevant planes are removed).
            std::vector<TGLPlane> activePlanes;
            for (planeInd=0; planeInd<maxPlanes; planeInd++)
            {
               activePlanes.push_back(planeSet[planeInd]);
               TGLPlane& p = activePlanes.back();
               p.Negate();
               glClipPlane(GL_CLIP_PLANE0+planeInd, p.CArr());
               glEnable(GL_CLIP_PLANE0+planeInd);

               // Draw scene with active planes, allocating fraction of time
               // for total planes.
               RenderElements(rnrCtx, elVec, check_timeout, &activePlanes);

               p.Negate();
               glClipPlane(GL_CLIP_PLANE0+planeInd, p.CArr());
            }
         }
         // Ensure all clip planes turned off again
         for (planeInd=0; planeInd<maxPlanes; planeInd++) {
            glDisable(GL_CLIP_PLANE0+planeInd);
         }
      }
   } // end for reqPasses

   // Reset gl modes to defaults
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glEnable(GL_CULL_FACE);
   glEnable(GL_LIGHTING);
}

//______________________________________________________________________________
void TGLScene::RenderElements(TGLRnrCtx&           rnrCtx,
                              DrawElementPtrVec_t& elVec,
                              Bool_t               check_timeout,
                              const TGLPlaneSet_t* clipPlanes)
{
   // Render DrawElements in elementVec with given timeout.
   // If clipPlanes is non-zero, test each element against its
   // clipping planes.

   TSceneInfo* sinfo = dynamic_cast<TSceneInfo*>(rnrCtx.GetSceneInfo());
   assert(sinfo != 0);

   Int_t drawCount = 0;

   for (DrawElementPtrVec_i i = elVec.begin(); i != elVec.end(); ++i)
   {
      const TGLPhysicalShape * drawShape = (*i)->fPhysical;

      Bool_t drawNeeded = kTRUE;

      // If clipping planes are passed as argument, we test against them.
      if (clipPlanes && IsOutside(drawShape->BoundingBox(), *clipPlanes))
         drawNeeded = kFALSE;

      // Draw?
      if (drawNeeded)
      {
         rnrCtx.SetShapeLOD((*i)->fFinalLOD);
         rnrCtx.SetShapePixSize((*i)->fPixelSize);
         glPushName(drawShape->ID());
         drawShape->Draw(rnrCtx);
         glPopName();
         ++drawCount;
         sinfo->UpdateDrawStats(*drawShape, rnrCtx.ShapeLOD());
      }

      // Terminate the draw if over opaque fraction timeout.
      // Only test every 2000 objects as this is somewhat costly.
      if (check_timeout && (drawCount % 2000) == 0 &&
          rnrCtx.HasStopwatchTimedOut())
      {
         if (rnrCtx.ViewerLOD() == TGLRnrCtx::kLODHigh)
             Warning("TGLScene::RenderElements",
                     "Timeout reached, not all elements rendered.");
         break;
      }
   }
}


/**************************************************************************/
// Selection
/**************************************************************************/

//______________________________________________________________________________
Bool_t TGLScene::ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx)
{
   // Process selection record rec.
   // 'curIdx' is the item position where the scene should start
   // its processing.
   // Return TRUE if an object has been identified or FALSE otherwise.
   // The scene-info member of the record is already set by the caller.

   if (curIdx >= rec.GetN())
      return kFALSE;

   TGLPhysicalShape* pshp = FindPhysical(rec.GetItem(curIdx));
   if (pshp)
   {
      rec.SetTransparent(pshp->IsTransparent());
      rec.SetPhysShape(pshp);
      rec.SetLogShape(const_cast<TGLLogicalShape*>(pshp->GetLogical()));
      rec.SetObject(pshp->GetLogical()->GetExternal());
      rec.SetSpecific(0);
      return kTRUE;
   }
   return kFALSE;
}


/**************************************************************************/
// Bounding-box
/**************************************************************************/

//______________________________________________________________________________
void TGLScene::CalcBoundingBox() const
{
   // Encapsulates all physical shapes bounding box with axes aligned box.
   // Validity checked in the base-class.

   Double_t xMin, xMax, yMin, yMax, zMin, zMax;
   xMin = xMax = yMin = yMax = zMin = zMax = 0.0;
   PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physicalShape;
   while (physicalShapeIt != fPhysicalShapes.end())
   {
      physicalShape = physicalShapeIt->second;
      if (!physicalShape)
      {
         assert(kFALSE);
         continue;
      }
      const TGLBoundingBox& box = physicalShape->BoundingBox();
      if (physicalShapeIt == fPhysicalShapes.begin()) {
         xMin = box.XMin(); xMax = box.XMax();
         yMin = box.YMin(); yMax = box.YMax();
         zMin = box.ZMin(); zMax = box.ZMax();
      } else {
         if (box.XMin() < xMin) { xMin = box.XMin(); }
         if (box.XMax() > xMax) { xMax = box.XMax(); }
         if (box.YMin() < yMin) { yMin = box.YMin(); }
         if (box.YMax() > yMax) { yMax = box.YMax(); }
         if (box.ZMin() < zMin) { zMin = box.ZMin(); }
         if (box.ZMax() > zMax) { zMax = box.ZMax(); }
      }
      ++physicalShapeIt;
   }
   fBoundingBox.SetAligned(TGLVertex3(xMin,yMin,zMin), TGLVertex3(xMax,yMax,zMax));
   fBoundingBoxValid = kTRUE;
}

/**************************************************************************/
// Logical shapes
/**************************************************************************/

//______________________________________________________________________________
void TGLScene::AdoptLogical(TGLLogicalShape& shape)
{
   // Adopt dynamically created logical 'shape' - add to internal map
   // and take responsibility for deleting.

   if (fLock != kModifyLock) {
      Error("TGLScene::AdoptLogical", "expected ModifyLock");
      return;
   }

   shape.fScene = this;
   fLogicalShapes.insert(LogicalShapeMapValueType_t(shape.ID(), &shape));
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyLogical(TObject* logid, Bool_t mustFind)
{
   // Destroy logical shape defined by unique 'ID'.
   // Returns kTRUE if found/destroyed - kFALSE otherwise.
   //
   // If mustFind is true, an error is reported if the logical is not
   // found.

   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyLogical", "expected ModifyLock");
      return kFALSE;
   }

   LogicalShapeMapIt_t lit = fLogicalShapes.find(logid);

   if (lit == fLogicalShapes.end()) {
      if (mustFind)
         Error("TGLScene::DestroyLogical", "logical not found in map.");
      return kFALSE;
   }

   TGLLogicalShape * logical = lit->second;
   UInt_t phid;
   while ((phid = logical->UnrefFirstPhysical()) != 0)
   {
      PhysicalShapeMapIt_t pit = fPhysicalShapes.find(phid);
      if (pit != fPhysicalShapes.end())
         DestroyPhysicalInternal(pit);
      else
         Warning("TGLScene::DestroyLogical", "an attached physical not found in map.");
   }
   assert(logical->Ref() == 0);
   fLogicalShapes.erase(lit);
   delete logical;
   InvalidateBoundingBox();
   IncTimeStamp();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TGLScene::DestroyLogicals()
{
   // Destroy all logical shapes in scene.
   // Return number of destroyed logicals.

   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyLogicals", "expected ModifyLock");
      return 0;
   }

   Int_t count = 0;
   LogicalShapeMapIt_t logicalShapeIt = fLogicalShapes.begin();
   const TGLLogicalShape * logicalShape;
   while (logicalShapeIt != fLogicalShapes.end()) {
      logicalShape = logicalShapeIt->second;
      if (logicalShape) {
         if (logicalShape->Ref() == 0) {
            fLogicalShapes.erase(logicalShapeIt++);
            delete logicalShape;
            ++count;
            continue;
         } else {
            assert(kFALSE);
         }
      } else {
         assert(kFALSE);
      }
      ++logicalShapeIt;
   }

   return count;
}

//______________________________________________________________________________
TGLLogicalShape * TGLScene::FindLogical(TObject* logid) const
{
   // Find and return logical shape identified by unqiue logid.
   // Returns 0 if not found.

   LogicalShapeMapCIt_t lit = fLogicalShapes.find(logid);
   if (lit != fLogicalShapes.end()) {
      return lit->second;
   } else {
      if (fInSmartRefresh)
         return FindLogicalSmartRefresh(logid);
      else
         return 0;
   }
}


/**************************************************************************/
// Physical shapes
/**************************************************************************/

//______________________________________________________________________________
void TGLScene::AdoptPhysical(TGLPhysicalShape & shape)
{
   // Adopt dynamically created physical 'shape' - add to internal map and take
   // responsibility for deleting
   if (fLock != kModifyLock) {
      Error("TGLScene::AdoptPhysical", "expected ModifyLock");
      return;
   }
   // TODO: Very inefficient check - disable
   assert(fPhysicalShapes.find(shape.ID()) == fPhysicalShapes.end());

   fPhysicalShapes.insert(PhysicalShapeMapValueType_t(shape.ID(), &shape));

   InvalidateBoundingBox();
   IncTimeStamp();
}

//______________________________________________________________________________
void TGLScene::DestroyPhysicalInternal(PhysicalShapeMapIt_t pit)
{
   // Virtual function to destroy a physical. Sub-classes might have
   // special checks to perform.
   // Caller should also invalidate the draw-list.

   delete pit->second;
   fPhysicalShapes.erase(pit);
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyPhysical(UInt_t phid)
{
   // Destroy physical shape defined by unique 'ID'.
   // Returns kTRUE if found/destroyed - kFALSE otherwise.

   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyPhysical", "expected ModifyLock.");
      return kFALSE;
   }

   PhysicalShapeMapIt_t pit = fPhysicalShapes.find(phid);

   if (pit == fPhysicalShapes.end()) {
      Error("TGLScene::DestroyPhysical::UpdatePhysical", "physical not found.");
      return kFALSE;
   }

   DestroyPhysicalInternal(pit);

   InvalidateBoundingBox();

   return kTRUE;
}

//______________________________________________________________________________
Int_t TGLScene::DestroyPhysicals()
{
   // Destroy physical shapes.

   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyPhysicals", "expected ModifyLock");
      return 0;
   }

   // Loop over logicals -- it is much more efficient that way.

   UInt_t count = 0;

   LogicalShapeMapIt_t lit = fLogicalShapes.begin();
   while (lit != fLogicalShapes.end())
   {
      TGLLogicalShape *lshp = lit->second;
      if (lshp && lshp->Ref() != 0)
      {
         count += lshp->Ref();
         lshp->DestroyPhysicals();
      }
      ++lit;
   }

   assert (count == fPhysicalShapes.size());
   fPhysicalShapes.clear();

   if (count > 0) {
      InvalidateBoundingBox();
      IncTimeStamp();
   }

   return count;
}

//______________________________________________________________________________
TGLPhysicalShape* TGLScene::FindPhysical(UInt_t phid) const
{
   // Find and return physical shape identified by unqiue 'ID'.
   // Returns 0 if not found.

   PhysicalShapeMapCIt_t pit = fPhysicalShapes.find(phid);
   return (pit != fPhysicalShapes.end()) ? pit->second : 0;
}

//______________________________________________________________________________
UInt_t TGLScene::GetMaxPhysicalID()
{
   // Returns the maximum used physical id.
   // Returns 0 if empty.

   if (fPhysicalShapes.empty()) return 0;
   return (--fPhysicalShapes.end())->first;
}


/**************************************************************************/
// Update methods
/**************************************************************************/

//______________________________________________________________________________
Bool_t TGLScene::BeginUpdate()
{
   // Put scene in update mode, return true if lock acquired.

   Bool_t ok = TakeLock(kModifyLock);
   return ok;
}

//______________________________________________________________________________
void TGLScene::EndUpdate(Bool_t minorChange, Bool_t sceneChanged, Bool_t updateViewers)
{
   // Exit scene update mode.
   //
   // If sceneChanged is true (default), the scene timestamp is
   // increased and basic draw-lists etc will be rebuild on next draw
   // request. If you only changed colors or some other visual
   // parameters that do not affect object bounding-box or
   // transformation matrix, you can set it to false.
   //
   // If updateViewers is true (default), the viewers using this scene
   // will be tagged as changed. If sceneChanged is true the
   // updateViewers should be true as well, unless you take care of
   // the viewers elsewhere or in some other way.

   if (minorChange)
      IncMinorStamp();

   if (sceneChanged)
      IncTimeStamp();

   ReleaseLock(kModifyLock);

   if (updateViewers)
      TagViewersChanged();
}

//______________________________________________________________________________
void TGLScene::UpdateLogical(TObject* logid)
{
   // Drop display-lists for the logical (assume TGLObject/direct rendering).
   // Re-calculate the bounding box (also for all physicals).

   if (fLock != kModifyLock) {
      Error("TGLScene::UpdateLogical", "expected ModifyLock");
      return;
   }

   TGLLogicalShape* log = FindLogical(logid);

   if (log == 0) {
      Error("TGLScene::UpdateLogical", "logical not found");
      return;
   }

   log->DLCacheClear();
   log->UpdateBoundingBox();
}

//______________________________________________________________________________
void TGLScene::UpdatePhysical(UInt_t phid, Double_t* trans, UChar_t* col)
{
   // Reposition/recolor physical shape.

   if (fLock != kModifyLock) {
      Error("TGLScene::UpdatePhysical", "expected ModifyLock");
      return;
   }

   TGLPhysicalShape* phys = FindPhysical(phid);

   if (phys == 0) {
      Error("TGLScene::UpdatePhysical", "physical not found");
      return;
   }

   if (trans)  phys->SetTransform(trans);
   if (col)    phys->SetDiffuseColor(col);
}

//______________________________________________________________________________
void TGLScene::UpdatePhysical(UInt_t phid, Double_t* trans, Color_t cidx, UChar_t transp)
{
   // Reposition/recolor physical shape.

   if (fLock != kModifyLock) {
      Error("TGLScene::UpdatePhysical", "expected ModifyLock");
      return;
   }

   TGLPhysicalShape* phys = FindPhysical(phid);

   if (phys == 0) {
      Error("TGLScene::UpdatePhysical", "physical not found");
      return;
   }

   if (trans)
      phys->SetTransform(trans);
   if (cidx >= 0) {
      Float_t rgba[4];
      RGBAFromColorIdx(rgba, cidx, transp);
      phys->SetDiffuseColor(rgba);
   }
}

//______________________________________________________________________________
void TGLScene::UpdatePhysioLogical(TObject* logid, Double_t* trans, UChar_t* col)
{
   // Reposition/recolor physical for given logical (assume TGLObject and
   // a single physical).

   if (fLock != kModifyLock) {
      Error("TGLScene::UpdatePhysioLogical", "expected ModifyLock");
      return;
   }

   TGLLogicalShape* log = FindLogical(logid);

   if (log == 0) {
      Error("TGLScene::UpdatePhysioLogical", "logical not found");
      return;
   }

   if (log->Ref() != 1) {
      Warning("TGLScene::UpdatePhysioLogical", "expecting a single physical (%d).", log->Ref());
   }

   TGLPhysicalShape* phys = log->fFirstPhysical;
   if (trans)  phys->SetTransform(trans);
   if (col)    phys->SetDiffuseColor(col);
}

//______________________________________________________________________________
void TGLScene::UpdatePhysioLogical(TObject* logid, Double_t* trans, Color_t cidx, UChar_t transp)
{
   // Reposition/recolor physical for given logical (assume TGLObject and
   // a single physical).

   if (fLock != kModifyLock) {
      Error("TGLScene::UpdatePhysioLogical", "expected ModifyLock");
      return;
   }

   TGLLogicalShape* log = FindLogical(logid);

   if (log == 0) {
      Error("TGLScene::UpdatePhysioLogical", "logical not found");
      return;
   }

   if (log->Ref() != 1) {
      Warning("TGLScene::UpdatePhysioLogical", "expecting a single physical (%d).", log->Ref());
   }

   TGLPhysicalShape* phys = log->fFirstPhysical;
   if (trans)
      phys->SetTransform(trans);
   if (cidx >= 0) {
      Float_t rgba[4];
      RGBAFromColorIdx(rgba, cidx, transp);
      phys->SetDiffuseColor(rgba);
   }
}


/**************************************************************************/
// Smart refresh
/**************************************************************************/

//______________________________________________________________________________
UInt_t TGLScene::BeginSmartRefresh()
{
   // Moves logicals that support smart-refresh to intermediate cache.
   // Destroys the others and returns the number of destroyed ones.

   fSmartRefreshCache.swap(fLogicalShapes);
   // Remove all logicals that don't survive a refresh.
   UInt_t count = 0;
   LogicalShapeMapIt_t i = fSmartRefreshCache.begin();
   while (i != fSmartRefreshCache.end()) {
      if (i->second->KeepDuringSmartRefresh() == kFALSE) {
         LogicalShapeMapIt_t j = i++;
         delete j->second;
         fSmartRefreshCache.erase(j);
         ++count;
      } else {
         ++i;
      }
   }
   fInSmartRefresh = kTRUE;
   return count;
}

//______________________________________________________________________________
void TGLScene::EndSmartRefresh()
{
   // Wipes logicals in refresh-cache.

   fInSmartRefresh = kFALSE;

   LogicalShapeMapIt_t i = fSmartRefreshCache.begin();
   while (i != fSmartRefreshCache.end()) {
      delete i->second;
      ++i;
   }
   fSmartRefreshCache.clear();
}

//______________________________________________________________________________
TGLLogicalShape * TGLScene::FindLogicalSmartRefresh(TObject* ID) const
{
   // Find and return logical shape identified by unqiue 'ID' in refresh-cache.
   // Returns 0 if not found.

   LogicalShapeMapIt_t it = fSmartRefreshCache.find(ID);
   if (it != fSmartRefreshCache.end())
   {
      TGLLogicalShape* l_shape = it->second;
      fSmartRefreshCache.erase(it);
      if (l_shape->IsA() != TGLObject::GetGLRenderer(ID->IsA()))
      {
         Warning("TGLScene::FindLogicalSmartRefresh", "Wrong renderer-type found in cache.");
         delete l_shape;
         return 0;
      }
      // printf("TGLScene::SmartRefresh found cached: %p '%s' [%s] for %p\n",
      //    l_shape, l_shape->GetExternal()->GetName(),
      //    l_shape->GetExternal()->IsA()->GetName(), (void*) ID);
      LogicalShapeMap_t* lsm = const_cast<LogicalShapeMap_t*>(&fLogicalShapes);
      lsm->insert(LogicalShapeMapValueType_t(l_shape->ID(), l_shape));
      l_shape->DLCacheClear();
      l_shape->UpdateBoundingBox();
      return l_shape;
   } else {
      return 0;
   }
}


/**************************************************************************/
// Helpers
/**************************************************************************/

//______________________________________________________________________________
UInt_t TGLScene::SizeOfScene() const
{
   // Return memory cost of scene.
   // Warning: NOT CORRECT at present - doesn't correctly calculate size.
   // of logical shapes with dynamic internal contents.

   UInt_t size = sizeof(*this);

   printf("Size: Scene Only %u\n", size);

   LogicalShapeMapCIt_t logicalShapeIt = fLogicalShapes.begin();
   const TGLLogicalShape * logicalShape;
   while (logicalShapeIt != fLogicalShapes.end()) {
      logicalShape = logicalShapeIt->second;
      size += sizeof(*logicalShape);
      ++logicalShapeIt;
   }

   printf("Size: Scene + Logical Shapes %u\n", size);

   PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physicalShape;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physicalShape = physicalShapeIt->second;
      size += sizeof(*physicalShape);
      ++physicalShapeIt;
   }

   printf("Size: Scene + Logical Shapes + Physical Shapes %u\n", size);

   return size;
}

//______________________________________________________________________________
void TGLScene::DumpMapSizes() const
{
   // Print sizes of logical nad physical-shape maps.

   printf("Scene: %u Logicals / %u Physicals\n",
          (UInt_t) fLogicalShapes.size(), (UInt_t) fPhysicalShapes.size());
}

//______________________________________________________________________________
void TGLScene::RGBAFromColorIdx(Float_t rgba[4], Color_t ci, Char_t transp)
{
   // Fill rgba color from ROOT color-index ci and transparency (0->100).

   TColor* c = gROOT->GetColor(ci);
   if(c)   c->GetRGB(rgba[0], rgba[1], rgba[2]);
   else    rgba[0] = rgba[1] = rgba[2] = 0.5;
   rgba[3] = 1.0f - transp/100.0f;
}

//______________________________________________________________________________
Bool_t TGLScene::IsOutside(const TGLBoundingBox & box,
                           const TGLPlaneSet_t  & planes)
{
   // Check if box is outside of all planes.

   for (TGLPlaneSet_ci p=planes.begin(); p!=planes.end(); ++p)
      if (box.Overlap(*p) == Rgl::kOutside)
         return kTRUE;
   return kFALSE;
}
