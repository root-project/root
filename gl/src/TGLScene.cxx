// @(#)root/gl:$Name:  $:$Id: TGLScene.cxx,v 1.43 2006/08/23 14:39:40 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original TGLRender by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TAttLine.h"
#include "TGLScene.h"
#include "TGLCamera.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLStopwatch.h"
#include "TGLDisplayListCache.h"
#include "TGLClip.h"
#include "TGLIncludes.h"
#include "TError.h"
#include "TString.h"
#include "TClass.h" // For non-TObject reflection
#include "TGLViewer.h" // Only here for some draw style enums - remove these
#include "TColor.h"     // moved to proper class
#include "TAtt3D.h"

#include <algorithm>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLScene                                                             //
//                                                                      //
// A GL scene is the container for all the viewable objects (shapes)    //
// loaded into the viewer. It consists of two main stl::maps containing //
// the TGLLogicalShape and TGLPhysicalShape collections, and interface  //
// functions enabling viewers to manage objects in these. The physical  //
// shapes defined the placement of copies of the logical shapes - see   //
// TGLLogicalShape/TGLPhysicalShape for more information on relationship//
//                                                                      //
// The scene can be drawn by owning viewer, passing camera, draw style  //
// & quality (LOD), clipping etc - see Draw(). The scene can also be    //
// drawn for selection in similar fashion - see Select(). The scene     //
// keeps track of a single selected physical - which can be modified by //
// viewers.                                                             //
//                                                                      //
// The scene maintains a lazy calculated bounding box for the total     //
// scene extents, axis aligned round TGLPhysicalShape shapes.           //
//                                                                      //
// Currently a scene is owned exclusively by one viewer - however it is //
// intended that it could easily be shared by multiple viewers - for    //
// efficiency and syncronisation reasons. Hence viewer variant objects  //
// camera, clips etc being owned by viewer and passed at draw/select    //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLScene)

//______________________________________________________________________________
TGLScene::TGLScene() :
   fLock(kUnlocked), fDrawList(1000),
   fDrawListValid(kFALSE),
   fInSmartRefresh(kFALSE),
   fBoundingBox(), fBoundingBoxValid(kFALSE),
   fSelectedPhysical(0),
   fSelectionResult(0),
   fNPrimHits(-1),
   fNSecHits(-1),
   fTrySecSelect(kFALSE),
   fTriedSecSelect(kFALSE),
   fSelectBuffer(4096),
   fSortedHits(),
   fClipPlane(0), fClipBox(0), fCurrentClip(0),
   fTransManip(), fScaleManip(), fRotateManip()
{
   // Construct scene object

   // Default manip is translation manipulator
   fCurrentManip = &fTransManip;
}

//______________________________________________________________________________
TGLScene::~TGLScene()
{
   // Purge out the DL cache - when per drawable DL purging implemented
   // this no longer really required. However should be faster....
   TGLDisplayListCache::Instance().Purge();

   // Delete clip objects
   ClearClips();

   // Destroy scene objects
   TakeLock(kModifyLock);
   DestroyPhysicals(kTRUE); // including modified
   DestroyLogicals();
   ReleaseLock(kModifyLock);
}

//TODO: Inline
//______________________________________________________________________________
void TGLScene::AdoptLogical(TGLLogicalShape & shape)
{
   // Adopt dynamically created logical 'shape' - add to internal map and take
   // responsibility for deleting
   if (fLock != kModifyLock) {
      Error("TGLScene::AdoptLogical", "expected ModifyLock");
      return;
   }

   // We can have mulitple instance of logical shapes with a zero ID
   // Very inefficient check - disabled
   /*if (shape.ID() != 0U) {
      assert(fLogicalShapes.find(shape.ID()) == fLogicalShapes.end());
   }*/
   fLogicalShapes.insert(LogicalShapeMapValueType_t(shape.ID(), &shape));
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyLogical(ULong_t ID)
{
   // Zero ID logical shapes are not unique - this simple means the external object
   // does not exist - should never be asked to destroy these singularly
   if (ID == 0U) {
      Error("TGLScene::DestroyLogical", "asked to destory non-unqiue 0 ID logical shape");
      return kFALSE;
   }

   // Destroy logical shape defined by unique 'ID'
   // Returns kTRUE if found/destroyed - kFALSE otherwise
   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyLogical", "expected ModifyLock");
      return kFALSE;
   }

   LogicalShapeMapIt_t logicalIt = fLogicalShapes.find(ID);
   if (logicalIt != fLogicalShapes.end()) {
      const TGLLogicalShape * logical = logicalIt->second;
      if (logical->Ref() == 0) {
         fLogicalShapes.erase(logicalIt);
         delete logical;
         return kTRUE;
      } else {
         assert(kFALSE);
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
UInt_t TGLScene::DestroyLogicals()
{
   // Destroy all logical shapes in scene
   // Return count of number destroyed
   UInt_t count = 0;
   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyLogicals", "expected ModifyLock");
      return count;
   }

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

//TODO: Inline
//______________________________________________________________________________
TGLLogicalShape * TGLScene::FindLogical(ULong_t ID) const
{
   // Zero ID logical shapes are not unique - this simple means the external object
   // does not exist - should never be asked to seach for this
   if (ID == 0U) {
      Error("TGLScene::FindLogical", "asked to find non-unqiue 0 ID logical shape");
      return 0;
   }

   // Find and return logical shape identified by unqiue 'ID'
   // Returns 0 if not found
   LogicalShapeMapCIt_t it = fLogicalShapes.find(ID);
   if (it != fLogicalShapes.end()) {
      return it->second;
   } else {
      if (fInSmartRefresh)
         return FindLogicalSmartRefresh(ID);
      else
         return 0;
   }
}

//TODO: Inline
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
   fBoundingBoxValid = kFALSE;

   // Add into draw list and mark for sorting
   fDrawList.push_back(&shape);
   fDrawListValid = kFALSE;
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyPhysical(ULong_t ID)
{
   // Destroy physical shape defined by unique 'ID'
   // Returns kTRUE if found/destroyed - kFALSE otherwise
   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyPhysical", "expected ModifyLock");
      return kFALSE;
   }
   PhysicalShapeMapIt_t physicalIt = fPhysicalShapes.find(ID);
   if (physicalIt != fPhysicalShapes.end()) {
      TGLPhysicalShape * physical = physicalIt->second;
      if (fSelectedPhysical == physical) {
         if (fCurrentManip->GetAttached() == fSelectedPhysical) {
            fCurrentManip->Attach(0);
         }
         fSelectedPhysical = 0;
      }
      fPhysicalShapes.erase(physicalIt);
      fBoundingBoxValid = kFALSE;

      // Zero the draw list entry - will be erased as part of sorting
      DrawListIt_t drawIt = find(fDrawList.begin(), fDrawList.end(), physical);
      if (drawIt != fDrawList.end()) {
         *drawIt = 0;
         fDrawListValid = kFALSE;
      } else {
         assert(kFALSE);
      }
      delete physical;
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
UInt_t TGLScene::DestroyPhysicals(Bool_t incModified, const TGLCamera * camera)
{
   // Destroy all logical shapes in scene
   // Return count of number destroyed
   if (fLock != kModifyLock) {
      Error("TGLScene::DestroyPhysicals", "expected ModifyLock");
      return kFALSE;
   }
   UInt_t count = 0;
   PhysicalShapeMapIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physical;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physical = physicalShapeIt->second;
      if (physical) {
         // Destroy any physical shape no longer of interest to camera
         // If modified options allow this physical to be destoyed
         if (incModified || (!incModified && !physical->IsModified())) {
            // and no camera is passed, or it is no longer of interest
            // to camera
            Bool_t ignoreSize = physical->GetLogical().IgnoreSizeForOfInterest();
            if (!camera || (camera && !camera->OfInterest(physical->BoundingBox(), ignoreSize))) {

               // Then we can destroy it - remove from map
               fPhysicalShapes.erase(physicalShapeIt++);

               // Zero the draw list entry - will be erased as part of sorting
               DrawListIt_t drawIt = find(fDrawList.begin(), fDrawList.end(), physical);
               if (drawIt != fDrawList.end()) {
                  *drawIt = 0;
               } else {
                  assert(kFALSE);
               }

               // Ensure if selected object this is cleared
               if (fSelectedPhysical == physical) {
                  if (fCurrentManip->GetAttached() == fSelectedPhysical) {
                     fCurrentManip->Attach(0);
                  }
                  fSelectedPhysical = 0;
               }
               // Finally destroy actual object
               delete physical;
               ++count;
               continue; // Incremented the iterator during erase()
            }
         }
      } else {
         assert(kFALSE);
      }
      ++physicalShapeIt;
   }

   if (count > 0) {
      fBoundingBoxValid = kFALSE;
      fDrawListValid = kFALSE;
   }

   return count;
}

//TODO: Inline
//______________________________________________________________________________
TGLPhysicalShape * TGLScene::FindPhysical(ULong_t ID) const
{
   // Find and return physical shape identified by unqiue 'ID'
   // Returns 0 if not found
   PhysicalShapeMapCIt_t it = fPhysicalShapes.find(ID);
   if (it != fPhysicalShapes.end()) {
      return it->second;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TGLScene::BeginSmartRefresh()
{
   // Moves logicals to refresh-cache.

   fSmartRefreshCache.swap(fLogicalShapes);
   // Remove all logicals that don't survive a refresh.
   LogicalShapeMapIt_t i = fSmartRefreshCache.begin();
   while (i != fSmartRefreshCache.end()) {
      if (i->second->KeepDuringSmartRefresh() == false) {
         delete i->second;
         fSmartRefreshCache.erase(i);
      }
      ++i;
   }
   fInSmartRefresh = true;
}

//______________________________________________________________________________
void TGLScene::EndSmartRefresh()
{
   // Wipes logicals in refresh-cache.

   fInSmartRefresh = false;

   LogicalShapeMapIt_t i = fSmartRefreshCache.begin();
   while (i != fSmartRefreshCache.end()) {
      delete i->second;
      ++i;
   }
   fSmartRefreshCache.clear();
}

TGLLogicalShape * TGLScene::FindLogicalSmartRefresh(ULong_t ID) const
{
   // Find and return logical shape identified by unqiue 'ID' in refresh-cache.
   // Returns 0 if not found.

   LogicalShapeMapIt_t it = fSmartRefreshCache.find(ID);
   if (it != fSmartRefreshCache.end()) {
      TGLLogicalShape* l_shape = it->second;
      fSmartRefreshCache.erase(it);
      // printf("TGLScene::SmartRefresh found cached: %p '%s' [%s] for %p\n",
      //    l_shape, l_shape->GetExternal()->GetName(),
      //    l_shape->GetExternal()->IsA()->GetName(), (void*) ID);
      LogicalShapeMap_t* lsm = const_cast<LogicalShapeMap_t*>(&fLogicalShapes);
      lsm->insert(LogicalShapeMapValueType_t(l_shape->ID(), l_shape));
      return l_shape;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TGLScene::Draw(const TGLCamera & camera, TGLDrawFlags sceneFlags,
                    Double_t timeout, Int_t axesType, const TGLVertex3 * reference,
                    Bool_t forSelect)
{
   // Draw out scene into current GL context, using passed arguments:
   //
   // 'camera' - used for for object culling, manip scalling
   // 'sceneFlags'  - draw flags for scene - see TGLDrawFlags
   // 'timeout'- timeout for scene draw (in milliseconds) - if 0.0 unlimited
   // 'axesType' - axis style - one of TGLViewer::EAxesType
   // 'reference' - position of reference marker (or none if null)
   // 'forSelect' - is draw for select? If kTRUE clip and manip objects (which
   //            cannot be selected) are not drawn

   // NOTE: We assume the following (set by calling TGLViewer)
   //
   // Suitible projection / modelview matricies - camera.Apply() called
   // In correct thread, with valid GL context current
   // Buffer swapping will be done after draw
   //
   // The camera is still passed for visibility culling, manip scaling etc
   if (fLock != kDrawLock && fLock != kSelectLock) {
      Error("TGLScene::Draw", "expected Draw or Select Lock");
   }

   // Reset debug draw stats
   ResetDrawStats(sceneFlags);

   // Sort the draw list if required
   if (!fDrawListValid) {
      SortDrawList();
   }

   // Setup GL light model and face culling depending on clip
   if (fCurrentClip) {
   } else {
   }

   // Setup GL for current draw style - fill, wireframe, outline
   // TODO: Could detect change and only mod if changed for speed
   UInt_t reqFullDraws = 1; // Default single full draw for fill+wireframe
   switch (sceneFlags.Style()) {
      case (TGLDrawFlags::kFill):
      case (TGLDrawFlags::kOutline): {
         glEnable(GL_LIGHTING);
         if (fCurrentClip) {
            // Clip object - two sided lighting, two side polygons, don't cull (BACK) faces
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glDisable(GL_CULL_FACE);
         }
         // No clip - default single side lighting,
         // front polygons, cull (BACK) faces ok
         if (sceneFlags.Style() == TGLDrawFlags::kOutline) {
            reqFullDraws = 2;   // Outline needs two full draws
         }

         break;
      }
      case (TGLDrawFlags::kWireFrame): {
         glDisable(GL_LIGHTING);
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         break;
      }
      default: {
         assert(kFALSE);
      }
   }

   for (UInt_t fullDraws = 0; fullDraws < reqFullDraws; fullDraws++) {
      // For outline two full draws (fill + wireframe) required.
      // Do it this way to avoid costly GL state swaps on per drawable basis
      if (sceneFlags.Style() == TGLDrawFlags::kOutline) {
         if (fullDraws == 0) {
            // First pass - offset polygons
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.f, 1.f);
         } else {
            // Second pass - outline (wireframe)
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
      }

      // Assume each full draw takes same time - probably too crude....
      Double_t fullDrawTimeout = timeout/reqFullDraws;

      // If no clip object no plane sets to extract/pass
      if (!fCurrentClip) {
         DrawPass(camera, sceneFlags, fullDrawTimeout);
      } else {
         // Get the clip plane set from the clipping object
         std::vector<TGLPlane> planeSet;
         fCurrentClip->PlaneSet(planeSet);

         // Strip any planes that outside the scene bounding box - no effect
         for (std::vector<TGLPlane>::iterator it = planeSet.begin();
              it != planeSet.end(); ) {
            if (BoundingBox().Overlap(*it) == kOutside) {
               it = planeSet.erase(it);
            } else {
               ++it;
            }
         }

         if (gDebug>2) {
            Info("TGLScene::Draw()", "%d active clip planes", planeSet.size());
         }
         // Limit to smaller of plane set size or GL implementation plane support
         Int_t maxGLPlanes;
         glGetIntegerv(GL_MAX_CLIP_PLANES, &maxGLPlanes);
         UInt_t maxPlanes = maxGLPlanes;
         UInt_t planeInd;
         if (planeSet.size() < maxPlanes) {
            maxPlanes = planeSet.size();
         }

         // Note : OpenGL Reference (Blue Book) states
         // GL_CLIP_PLANEi = CL_CLIP_PLANE0 + i

         // Clip away scene outside of the clip object
         if (fCurrentClip->Mode() == TGLClip::kOutside) {
            // Load all negated clip planes (up to max) at once
            for (UInt_t i=0; i<maxPlanes; i++) {
               planeSet[i].Negate();
               glClipPlane(GL_CLIP_PLANE0+i, planeSet[i].CArr());
               glEnable(GL_CLIP_PLANE0+i);
            }

             // Draw scene once with full time slot, passing all the planes
             DrawPass(camera, sceneFlags, fullDrawTimeout, &planeSet);
         }
         // Clip away scene inside of the clip object
         else {
            std::vector<TGLPlane> activePlanes;
            for (planeInd=0; planeInd<maxPlanes; planeInd++) {
               if (planeInd > 0) {
                  activePlanes[planeInd - 1].Negate();
                  glClipPlane(GL_CLIP_PLANE0+planeInd - 1, activePlanes[planeInd - 1].CArr());
               }
               activePlanes.push_back(planeSet[planeInd]);
               glClipPlane(GL_CLIP_PLANE0+planeInd, activePlanes[planeInd].CArr());
               glEnable(GL_CLIP_PLANE0+planeInd);

               // Draw scene with active planes, allocating fraction of time
               // for total planes.
               DrawPass(camera, sceneFlags, fullDrawTimeout/maxPlanes, &activePlanes);
            }
         }
         // Ensure all clip planes turned off again
         for (planeInd=0; planeInd<maxPlanes; planeInd++) {
            glDisable(GL_CLIP_PLANE0+planeInd);
         }
      }
   }

   // Reset gl modes to defaults
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
   glPolygonMode(GL_FRONT, GL_FILL);
   glEnable(GL_CULL_FACE);
   glEnable(GL_LIGHTING);

   // Draw guides - must be done before manipulator / selected object
   // bounding box as we clear the depth buffer
   DrawGuides(camera, axesType, reference);

   // If select draw clip and manips are not drawn (pickable)
   if (!forSelect) {
      // Draw the clip shape (unclipped!) if it is being manipulated
      if (fCurrentClip && fCurrentManip->GetAttached() == fCurrentClip) {
         // Clip objects are always drawn with normal fill style
         // regardless of main scene objects draw style
         TGLDrawFlags clipSceneFlags(TGLDrawFlags::kFill, sceneFlags.LOD());
         fCurrentClip->Draw(fCurrentClip->CalcDrawFlags(camera, clipSceneFlags));
      }

      // Draw the current manipulator - we want it depth buffer clipped against itself
      // but not the rest of the scene (so it appears over them)
      glClear(GL_DEPTH_BUFFER_BIT);
      fCurrentManip->Draw(camera);

      // Draw selected object bounding box
      if (fSelectedPhysical) {
         if (sceneFlags.Style() == TGLDrawFlags::kFill ||
             sceneFlags.Style() == TGLDrawFlags::kWireFrame) {
            // White for wireframe and fill style,
            glColor3d(1.0, 1.0, 1.0);
         } else {
            // Red for outlines
            glColor3d(1.0, 0.0, 0.0);
         }
         if (sceneFlags.Style() == TGLDrawFlags::kFill ||
             sceneFlags.Style() == TGLDrawFlags::kOutline) {
            glDisable(GL_LIGHTING);
         }
         fSelectedPhysical->BoundingBox().Draw();
         if (sceneFlags.Style() == TGLDrawFlags::kFill ||
             sceneFlags.Style() == TGLDrawFlags::kOutline) {
            glEnable(GL_LIGHTING);
         }
      }
   }

   // Dump debug draw stats
   DumpDrawStats();
}

//______________________________________________________________________________
void TGLScene::DrawPass(const TGLCamera & camera, const TGLDrawFlags & sceneFlags,
                        Double_t timeout, const std::vector<TGLPlane> * clipPlanes)
{
   // Perform a internal draw pass - multiple passes are required for some
   // clip shapes

   // 'camera' - used for for object culling
   // 'style'  - draw style kFill (filled polygons) kOutline (polygons + outlines)
   //            kWireFrame
   // 'LOD'    - base scene level of detail (quality), value 0 (low) to 100 (high)
   //            combined with projection size LOD to produce overall draw LOD
   //            for each physical shape
   // 'timeout'- timeout for pass (in milliseconds) - if 0.0 unlimited
   // 'clipPlanes' - collection of active clip planes - used for further culling
   //

   // NOTE: We do not apply clip planes (at GL level) in this function - this is
   // already done in Draw(). The clipPlanes is passed only for shape culling
   // before the camera cull is done

   // Set stopwatch running
   TGLStopwatch stopwatch;
   if (timeout > 0.0 || gDebug > 2) {
      stopwatch.Start();
   }

   // Step 1: Loop through the main sorted draw list
   Bool_t                   run = kTRUE;
   const TGLPhysicalShape * drawShape;
   Bool_t                   doSelected = (fSelectedPhysical != 0);

   // Transparent list built on fly
   static DrawList_t transDrawList;
   transDrawList.reserve(fDrawList.size() / 10); // assume less 10% of total
   transDrawList.clear();

   // Opaque only objects drawn in first loop - transparent ones
   // which require drawing added to list during this
   // TODO: Sort front -> back for better performance
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);

   // If the scene bounding box is inside the camera frustum then
   // no need to check individual shapes - everything is visible
   Bool_t useFrustumCheck = (camera.FrustumOverlap(BoundingBox()) != kInside);

   TGLDrawFlags shapeFlags;
   DrawListIt_t drawIt;
   for (drawIt = fDrawList.begin(); drawIt != fDrawList.end() && run;
        ++drawIt) {
      drawShape = *drawIt;
      if (!drawShape)
      {
         assert(kFALSE);
         continue;
      }

      // Selected physical should always be drawn (and only once) if visible,
      // regardless of time limited draws
      if (drawShape == fSelectedPhysical) {
         doSelected = kFALSE;
      }

      // TODO: Do small skipping first? Probably cheaper than frustum check
      // Profile relative costs? The frustum check could be done implictly
      // from the LOD as we project all 8 verticies of the BB onto viewport

      // Work out if we need to draw this shape - assume we do first
      Bool_t drawNeeded = kTRUE;
      EOverlap overlap;

      // Draw test against passed clipping planes
      // Do before camera clipping on assumption clip planes remove more objects
      if (clipPlanes) {
         for (UInt_t i = 0; i < clipPlanes->size(); ++i) {
            overlap = drawShape->BoundingBox().Overlap((*clipPlanes)[i]);
            if (overlap == kOutside) {
               drawNeeded = kFALSE;
               break;
            }
         }
      }

      // Draw test against camera frustum if require
      if (drawNeeded && useFrustumCheck)
      {
         overlap = camera.FrustumOverlap(drawShape->BoundingBox());
         drawNeeded = overlap == kInside || overlap == kPartial;
      }

      // Draw?
      if (drawNeeded)
      {
         // Defer transparent shape drawing to after opaque
         if (drawShape->IsTransparent()) {
            transDrawList.push_back(drawShape);
            continue;
         }

         shapeFlags = drawShape->CalcDrawFlags(camera, sceneFlags);
         drawShape->Draw(shapeFlags);
         UpdateDrawStats(*drawShape, shapeFlags);
      }

      // Terminate the draw if over opaque fraction timeout
      // Only test every 50 objects as this is somewhat costly
      if (timeout > 0.0 && fDrawStats.fOpaque > 0 && (fDrawStats.fOpaque % 50) == 0) {
         Double_t opaqueTimeFraction = static_cast<Double_t>(fDrawStats.fOpaque) /
                                       static_cast<Double_t>(transDrawList.size() + fDrawStats.fOpaque);
         if (stopwatch.Lap() > (timeout * opaqueTimeFraction)) {
            run = kFALSE;
         }
      }
   }

   // Step 2: Deal with selected physical in case skipped by timeout of above loop
   if (doSelected) {
      // Draw now if non-transparent
      if (!fSelectedPhysical->IsTransparent()) {
         shapeFlags = fSelectedPhysical->CalcDrawFlags(camera, sceneFlags);
         fSelectedPhysical->Draw(shapeFlags);
         UpdateDrawStats(*fSelectedPhysical, shapeFlags);
      } else {
         // Add to transparent drawlist
         transDrawList.push_back(fSelectedPhysical);
      }
   }

   // Step 3: Draw the filtered transparent objects with GL depth writing off
   // blending on
   // TODO: Sort to draw back to front with depth test off for better blending
   glDepthMask(GL_FALSE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   for (drawIt = transDrawList.begin(); drawIt != transDrawList.end(); drawIt++) {
      drawShape = *drawIt;
         shapeFlags = drawShape->CalcDrawFlags(camera, sceneFlags);
         drawShape->Draw(shapeFlags);
         UpdateDrawStats(*drawShape, shapeFlags);
   }

   // Reset these after transparent done
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);

   if (gDebug > 2) {
      Info("TGLScene::DrawPass", "requested in %f msec, took %f msec", timeout, stopwatch.End());
   }
}

//______________________________________________________________________________
void TGLScene::SortDrawList()
{
   // Sort the TGLPhysical draw list by shape bounding box volume, from
   // large to small. This makes dropout of shapes with time limited
   // Draw() calls must less noticable. As this does not use projected
   // size it only needs to be done after a scene content change - not
   // everytime scene drawn (potential camera/projection change).
   assert(!fDrawListValid);

   TGLStopwatch stopwatch;

   if (gDebug>2) {
      stopwatch.Start();
   }

   fDrawList.reserve(fPhysicalShapes.size());

   // Delete all zero (to-be-deleted) objects
   fDrawList.erase(remove(fDrawList.begin(), fDrawList.end(), static_cast<const TGLPhysicalShape *>(0)), fDrawList.end());

   assert(fDrawList.size() == fPhysicalShapes.size());

   //TODO: partition the selected to front

   // Sort by volume of shape bounding box
   sort(fDrawList.begin(), fDrawList.end(), TGLScene::ComparePhysicalVolumes);

   if (gDebug>2) {
      Info("TGLScene::SortDrawList", "sorting took %f msec", stopwatch.End());
   }

   fDrawListValid = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLScene::ComparePhysicalVolumes(const TGLPhysicalShape * shape1, const TGLPhysicalShape * shape2)
{
   // Compare 'shape1' and 'shape2' bounding box volumes - return kTRUE if
   // 'shape1' bigger than 'shape2'

   // TODO: Move this to TGLBoundingBox > operator?
   return (shape1->BoundingBox().Volume() > shape2->BoundingBox().Volume());
}

//______________________________________________________________________________
void TGLScene::DrawGuides(const TGLCamera & camera, Int_t axesType, const TGLVertex3 * reference) const
{
   // Draw out scene guides - axes and reference marker
   //
   // 'camera'    - current active camera - required for calculating projection size
   // 'axesType'  - kAxesNone, kAxesOrigin (run through origin), kAxesEdge (at scene box edge)
   // 'reference' - if not null, draw orange reference sphere at vertex
   if (fLock != kDrawLock && fLock != kSelectLock) {
      Error("TGLScene::DrawMarkers", "expected Draw or Select Lock");
   }

   // Reference and origin based axes are not depth clipped
   glDisable(GL_DEPTH_TEST);

   // Draw any passed reference marker
   if (reference) {
      const Float_t referenceColor[4] = { 0.98, 0.45, 0.0, 1.0 }; // Orange
      TGLVector3 referenceSize = camera.ViewportDeltaToWorld(*reference, 3, 3);
      TGLUtil::DrawSphere(*reference, referenceSize.Mag(), referenceColor);
   }

   if (axesType != TGLViewer::kAxesOrigin) {
      glEnable(GL_DEPTH_TEST);
   }
   if (axesType == TGLViewer::kAxesNone) {
      return;
   }

   const Float_t axesColors[][4] = {{0.5, 0.0, 0.0, 1.0},  // -ive X axis light red
                                    {1.0, 0.0, 0.0, 1.0},  // +ive X axis deep red
                                    {0.0, 0.5, 0.0, 1.0},  // -ive Y axis light green
                                    {0.0, 1.0, 0.0, 1.0},  // +ive Y axis deep green
                                    {0.0, 0.0, 0.5, 1.0},  // -ive Z axis light blue
                                    {0.0, 0.0, 1.0, 1.0}}; // +ive Z axis deep blue


   // Axes draw at fixed screen size - back project to world
   TGLVector3 pixelVector = camera.ViewportDeltaToWorld(BoundingBox().Center(), 1, 1);
   Double_t pixelSize = pixelVector.Mag();

   // Find x/y/z min/max values
   Double_t min[3] = { BoundingBox().XMin(), BoundingBox().YMin(), BoundingBox().ZMin() };
   Double_t max[3] = { BoundingBox().XMax(), BoundingBox().YMax(), BoundingBox().ZMax() };

   for (UInt_t i = 0; i < 3; i++) {
      TGLVertex3 start;
      TGLVector3 vector;

      if (axesType == TGLViewer::kAxesOrigin) {
         // Through origin axes
         start[(i+1)%3] = 0.0;
         start[(i+2)%3] = 0.0;
      } else {
         // Side axes
         start[(i+1)%3] = min[(i+1)%3];
         start[(i+2)%3] = min[(i+2)%3];
      }
      vector[(i+1)%3] = 0.0;
      vector[(i+2)%3] = 0.0;

      // -ive axis?
      if (min[i] < 0.0) {
         // Runs from origin?
         if (max[i] > 0.0) {
            start[i] = 0.0;
            vector[i] = min[i];
         } else {
            start[i] = max[i];
            vector[i] = min[i] - max[i];
         }
         TGLUtil::DrawLine(start, vector, TGLUtil::kLineHeadNone, pixelSize*2.5, axesColors[i*2]);
      }
      // +ive axis?
      if (max[i] > 0.0) {
         // Runs from origin?
         if (min[i] < 0.0) {
            start[i] = 0.0;
            vector[i] = max[i];
         } else {
            start[i] = min[i];
            vector[i] = max[i] - min[i];
         }
         TGLUtil::DrawLine(start, vector, TGLUtil::kLineHeadNone, pixelSize*2.5, axesColors[i*2 + 1]);
      }
   }

   // Draw origin sphere(s)
   if (axesType == TGLViewer::kAxesOrigin) {
      // Single white origin sphere at 0, 0, 0
      Float_t white[4] = { 1.0, 1.0, 1.0, 1.0 };
      TGLUtil::DrawSphere(TGLVertex3(0.0, 0.0, 0.0), pixelSize*2.0, white);
   } else {
      for (UInt_t j = 0; j < 3; j++) {
         if (min[j] <= 0.0 && max[j] >= 0.0) {
            TGLVertex3 zero;
            zero[j] = 0.0;
            zero[(j+1)%3] = min[(j+1)%3];
            zero[(j+2)%3] = min[(j+2)%3];
            TGLUtil::DrawSphere(zero, pixelSize*2.0, axesColors[j*2 + 1]);
         }
      }
   }

   static const UChar_t xyz[][8] = {{0x44, 0x44, 0x28, 0x10, 0x10, 0x28, 0x44, 0x44},
                                    {0x10, 0x10, 0x10, 0x10, 0x10, 0x28, 0x44, 0x44},
                                    {0x7c, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x7c}};

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   // Labels
   Double_t padPixels = 25.0;

   glDisable(GL_LIGHTING);
   for (UInt_t k = 0; k < 3; k++) {
      TGLUtil::SetDrawColors(axesColors[k*2+1]);
      TGLVertex3 minPos, maxPos;
      if (axesType == TGLViewer::kAxesOrigin) {
         minPos[(k+1)%3] = 0.0;
         minPos[(k+2)%3] = 0.0;
      } else {
         minPos[(k+1)%3] = min[(k+1)%3];
         minPos[(k+2)%3] = min[(k+2)%3];
      }
      maxPos = minPos;
      minPos[k] = min[k];
      maxPos[k] = max[k];

      TGLVector3 axis = maxPos - minPos;
      TGLVector3 axisViewport = camera.WorldDeltaToViewport(minPos, axis);

      // Skip drawning if viewport projection of axis very small - labels will overlap
      // Occurs with orthographic cameras
      if (axisViewport.Mag() < 1) {
         continue;
      }

      minPos -= camera.ViewportDeltaToWorld(minPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                                    padPixels*axisViewport.Y()/axisViewport.Mag());
      axisViewport = camera.WorldDeltaToViewport(maxPos, -axis);
      maxPos -= camera.ViewportDeltaToWorld(maxPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                                    padPixels*axisViewport.Y()/axisViewport.Mag());

      DrawNumber(min[k], minPos);        // Min value
      DrawNumber(max[k], maxPos);        // Max value

      // Axis name beside max value
      TGLVertex3 namePos = maxPos -
         camera.ViewportDeltaToWorld(maxPos, padPixels*axisViewport.X()/axisViewport.Mag(),
                                     padPixels*axisViewport.Y()/axisViewport.Mag());
      glRasterPos3dv(namePos.CArr());
      glBitmap(8, 8, 0.0, 4.0, 0.0, 0.0, xyz[k]); // Axis Name
   }
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
}

//______________________________________________________________________________
void TGLScene::DrawNumber(Double_t num, const TGLVertex3 & center) const
{
   // Draw out number (as string) 'num', centered on vertex 'center'
   if (fLock != kDrawLock && fLock != kSelectLock) {
      Error("TGLScene::DrawNumber", "expected Draw or Select Lock");
   }
   static const UChar_t
      digits[][8] = {{0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38},//0
                     {0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x10, 0x10},//1
                     {0x7c, 0x44, 0x20, 0x18, 0x04, 0x04, 0x44, 0x38},//2
                     {0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x44, 0x38},//3
                     {0x04, 0x04, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x44},//4
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x40, 0x40, 0x7c},//5
                     {0x7c, 0x44, 0x44, 0x44, 0x7c, 0x40, 0x40, 0x7c},//6
                     {0x20, 0x20, 0x20, 0x10, 0x08, 0x04, 0x44, 0x7c},//7
                     {0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38},//8
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x7c},//9
                     {0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},//.
                     {0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00}};//-

   TString str;
   str+=Long_t(num);
   Double_t xOffset = 3.5 * str.Length();
   Double_t yOffset = 4.0;
   glRasterPos3dv(center.CArr());
   for (Ssiz_t i = 0, e = str.Length(); i < e; ++i) {
      if (str[i] == '.') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[10]);
      } else if (str[i] == '-') {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[11]);
      } else {
         glBitmap(8, 8, xOffset, yOffset, 7.0, 0.0, digits[str[i] - '0']);
      }
   }
}

//______________________________________________________________________________
Bool_t TGLScene::Select(const TGLCamera & camera, const TGLDrawFlags & sceneFlags)
{
   // Perform select draw using arguments:
   //
   // 'camera' - used for for object culling
   // 'style'  - draw style kFill (filled polygons) kOutline (polygons + outlines)
   //            kWireFrame
   //
   // Arguments are passed on to Draw(), with unlimted time
   //
   // Returns kTRUE if selected object is different then currently selected
   // fSelectedPhysical. The result is stored in fSelectionResult member.
   // One still has to call ApplySelection() to activate this change.
   //
   // If secondary-selection is activated (by calling
   // ActivateSecSelect() prior requesting the selection another
   // selection pass is done on selected object if it supports
   // secondary selection. The secondary select records are then available in:
   // fTriedSecSelect, fNSecHits, fSelectBuffer and fSortedHits.

   Bool_t changed = kFALSE;
   if (fLock != kSelectLock) {
      Error("TGLScene::Select", "expected SelectLock");
   }

   fSelectionResult =  0;
   fNPrimHits       = -1;
   fNSecHits        = -1;
   fTriedSecSelect  =  kFALSE;
   Bool_t trySecSel =  fTrySecSelect;
   fTrySecSelect    =  kFALSE;

   // Check size of the select buffer. This will work as we have a flat set of physical shapes.
   // We only ever load a single name in TGLPhysicalShape::DirectDraw so any hit record always
   // has same 4 GLuint format
   if (fPhysicalShapes.size()*4 > fSelectBuffer.size())
      fSelectBuffer.resize(fPhysicalShapes.size()*4);
   glSelectBuffer(fSelectBuffer.size(), &fSelectBuffer[0]);

   TGLDrawFlags selFlags(sceneFlags.Style(), sceneFlags.LOD());
   selFlags.SetSelection(kTRUE);

   // Enter picking mode
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);

   // Draw out scene at best quality, no timelimit, no axes/reference
   Draw(camera, selFlags, 0.0, TGLViewer::kAxesNone, 0, kTRUE); // Select draw

   // Retrieve the hit count and return to render
   fNPrimHits = glRenderMode(GL_RENDER);

   if (fNPrimHits < 0) {
      Error("TGLScene::Select", "selection buffer overflow");
      return changed;
   }

   if (fNPrimHits > 0) {
      // Every hit record has format (GLuint per item) - format is:
      //
      // no of names in name block (1 always)
      // minDepth
      // maxDepth
      // name(s) (1 always)

      // Sort the hits by minimum depth (closest part of object)
      fSortedHits.resize(fNPrimHits);
      Int_t i;
      for (i = 0; i < fNPrimHits; i++) {
         assert(fSelectBuffer[i * 4] == 1); // expect a single name per record
         fSortedHits[i].first  =  fSelectBuffer[4*i + 1]; // hit object minimum depth
         fSortedHits[i].second = &fSelectBuffer[4*i];     // hit record
      }
      std::sort(fSortedHits.begin(), fSortedHits.end());

      // Find first (closest) non-transparent object in the hit stack
      for (i = 0; i < fNPrimHits; i++) {
         fSelectionResult = FindPhysical(fSortedHits[i].second[3]);
         if (!fSelectionResult->IsTransparent()) {
            break;
         }
      }
      // If we failed to find a non-transparent object use the first
      // (closest) transparent one
      if (fSelectionResult->IsTransparent()) {
         fSelectionResult = FindPhysical(fSortedHits[0].second[3]);
      }
      assert(fSelectionResult);

      // Swap any selection
      if (fSelectionResult != fSelectedPhysical) {
         changed = kTRUE;
      }


      TGLLogicalShape& log = const_cast<TGLLogicalShape&>(fSelectionResult->GetLogical());
      if(trySecSel && log.SupportsSecondarySelect()) {
         fTriedSecSelect = kTRUE;

         DrawList_t  drawlist_tmp;
         drawlist_tmp.push_back(fSelectionResult);
         fDrawList.swap(drawlist_tmp);

         glRenderMode(GL_SELECT);
         glInitNames();
         glPushName(0);
         selFlags.SetSecSelection(kTRUE);
         Draw(camera, selFlags, 0.0, TGLViewer::kAxesNone, 0, kTRUE); 
         fNSecHits = glRenderMode(GL_RENDER);

         fDrawList.swap(drawlist_tmp);

         if (fNSecHits < 0) {
            Error("TGLScene::Select", "selection buffer overflow");
            return changed;
         }

         if (fNSecHits > 0) {
            fSortedHits.resize(fNSecHits);

            UInt_t *ptr = &fSelectBuffer[0];
            for (Int_t i = 0; i < fNSecHits; i++) {
               fSortedHits[i].first  = ptr[1]; // hit object minimum depth
               fSortedHits[i].second = ptr;    // pointer to hit entry 
               // printf("hit %d,  z1=%u, z2=%u, names=%u\n", i, ptr[1], ptr[2], ptr[0]); 
               ptr += 3 + ptr[0];
            }

            std::sort(fSortedHits.begin(), fSortedHits.end());
         }
      }

   } else { // 0 prim-hits
      if (fSelectedPhysical || fCurrentManip->GetAttached()) {
         changed = kTRUE;
      }
   }

   return changed;
}

//______________________________________________________________________________
void TGLScene::ApplySelection()
{
   // Makes result of last selection (fSelectionResult) the currently
   // selected object (fSelectedPhysical).

   if (fSelectedPhysical) {
      fSelectedPhysical->Select(kFALSE);
   }
   fSelectedPhysical = fSelectionResult;
   if (fSelectedPhysical) {
      fSelectedPhysical->Select(kTRUE);
   }
   // Always attach current manipulator as manip can be swapped to clip object
   fCurrentManip->Attach(fSelectedPhysical);
}

//______________________________________________________________________________
Bool_t TGLScene::SetSelectedColor(const Float_t color[17])
{
   // Set full color attributes on current selected physical shape:
   //
   // 0...3  - diffuse
   // 4...7  - ambient
   // 8...11 - specular
   // 12..15 - emission
   // 16     - shininess
   //
   // see OpenGL documentation for details of materials
   if (fSelectedPhysical) {
      fSelectedPhysical->SetColor(color);
      return kTRUE;
   } else {
      assert(kFALSE);
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLScene::SetColorOnSelectedFamily(const Float_t color[17])
{
   // Set full color attributes on all physical shapes sharing the same
   // logical shape as the selected physical
   //
   // 0...3  - diffuse
   // 4...7  - ambient
   // 8...11 - specular
   // 12..15 - emission
   // 16     - shininess
   //
   // see OpenGL documentation for details of materials
   if (fSelectedPhysical) {
      if (TAttLine *test = dynamic_cast<TAttLine *>(fSelectedPhysical->GetLogical().GetExternal()))
         test->SetLineColor(TColor::GetColor(color[0], color[1], color[2]));

      TGLPhysicalShape * physical;
      PhysicalShapeMapIt_t physicalShapeIt = fPhysicalShapes.begin();
      while (physicalShapeIt != fPhysicalShapes.end()) {
         physical = physicalShapeIt->second;
         if (physical) {
            if (physical->GetLogical().ID() == fSelectedPhysical->GetLogical().ID()) {
               physical->SetColor(color);
            }
         } else {
            assert(kFALSE);
         }
         ++physicalShapeIt;
      }
      return kTRUE;
   } else {
      assert(kFALSE);
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLScene::SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale)
{
   // Update geometry of the selected physical. 'trans' and 'scale' specify the
   // translation and scaling components of the physical shapes translation matrix
   // See TGLMatrix for more details
   if (fSelectedPhysical) {
      fSelectedPhysical->SetTranslation(trans);
      fSelectedPhysical->Scale(scale);
      fBoundingBoxValid = kFALSE;
      return kTRUE;
   } else {
      assert(kFALSE);
      return kFALSE;
   }
}

//______________________________________________________________________________
void TGLScene::SetupClips()
{
   // Setup clipping objects for current scene bounding box

   // Clear out any previous clips
   ClearClips();

   fClipPlane = new TGLClipPlane(TGLVector3(0.0,-1.0,0.0),
                                 BoundingBox().Center(),
                                 BoundingBox().Extents().Mag()*5.0);

   TGLVector3 halfLengths = BoundingBox().Extents() * 0.2501;
   TGLVertex3 center = BoundingBox().Center() + halfLengths;
   fClipBox = new TGLClipBox(halfLengths, center);
}

//______________________________________________________________________________
void TGLScene::ClearClips()
{
   // Clear out exising clipping objects
   if (fCurrentManip->GetAttached() == fCurrentClip) {
      fCurrentManip->Attach(0);
   }
   fCurrentClip = 0;

   delete fClipPlane;
   delete fClipBox;
}

//______________________________________________________________________________
void TGLScene::GetClipState(EClipType type, Double_t data[6]) const
{
   // Get state of clip object 'type' into data vector:
   //
   // 'type' requested        'data' contents returned
   // kClipPlane              4 components - A,B,C,D - of plane eq : Ax+By+CZ+D = 0
   // kBoxPlane               6 components - Box Center X/Y/Z - Box Extents X/Y/Z
   if (type == kClipPlane) {
      TGLPlaneSet_t planes;
      fClipPlane->PlaneSet(planes);
      data[0] = planes[0].A();
      data[1] = planes[0].B();
      data[2] = planes[0].C();
      data[3] = planes[0].D();
   } else if (type == kClipBox) {
      const TGLBoundingBox & box = fClipBox->BoundingBox();
      TGLVector3 ext = box.Extents();
      data[0] = box.Center().X();
      data[1] = box.Center().Y();
      data[2] = box.Center().Z();
      data[3] = box.Extents().X();
      data[4] = box.Extents().Y();
      data[5] = box.Extents().Z();
   } else {
      Error("TGLScene::GetClipState", "invalid clip type");
   }
}

//______________________________________________________________________________
void TGLScene::SetClipState(EClipType type, const Double_t data[6])
{
   // Set state of clip object 'type' into data vector:
   //
   // 'type' specified        'data' contents interpretation
   // kClipNone               ignored
   // kClipPlane              4 components - A,B,C,D - of plane eq : Ax+By+CZ+D = 0
   // kBoxPlane               6 components - Box Center X/Y/Z - Box Extents X/Y/Z
   switch (type) {
      case(kClipNone): {
         break;
      }
      case(kClipPlane): {
         TGLPlane newPlane(data[0], data[1], data[2], data[3]);
         fClipPlane->Set(newPlane);
         break;
      }
      case(kClipBox): {
         //TODO: Pull these inside TGLPhysicalShape
         // Update clip box center
         const TGLBoundingBox & currentBox = fClipBox->BoundingBox();
         TGLVector3 shift(data[0] - currentBox.Center().X(),
                          data[1] - currentBox.Center().Y(),
                          data[2] - currentBox.Center().Z());
         fClipBox->Translate(shift);
         // Update clip box extents

         TGLVector3 currentScale = fClipBox->GetScale();
         TGLVector3 newScale(data[3] / currentBox.Extents().X() * currentScale.X(),
                             data[4] / currentBox.Extents().Y() * currentScale.Y(),
                             data[5] / currentBox.Extents().Z() * currentScale.Z());

         fClipBox->Scale(newScale);
         break;
      }
   }
}

//______________________________________________________________________________
void TGLScene::GetCurrentClip(EClipType & type, Bool_t & edit) const
{
   // Get current type active in viewer - returns one of kClipNone
   // kClipPlane or kClipBox
   if (fCurrentClip == 0) {
      type = kClipNone;
      edit = kFALSE;
   } else if (fCurrentClip == fClipPlane) {
      type = kClipPlane;
      edit = (fCurrentManip->GetAttached() == fCurrentClip);
   } else if (fCurrentClip == fClipBox) {
      type = kClipBox;
      edit = (fCurrentManip->GetAttached() == fCurrentClip);
   } else {
      Error("TGLScene::GetCurrentClip" , "Unknown clip type");
      type = kClipNone;
      edit = kFALSE;
   }
}

//______________________________________________________________________________
void TGLScene::SetCurrentClip(EClipType type, Bool_t edit)
{
   // Set current clip active in viewer - 'type' is one of kClipNone
   // kClipPlane or kClipBox. 'edit' indicates if clip object should
   // been shown/edited directly in viewer (current manipulator attached to it)
   // kTRUE if so (ignored for kClipNone), kFALSE otherwise

   // If edit being turned off and current manip is attached to
   // current clip detach it
   if (!edit && fCurrentManip->GetAttached() == fCurrentClip) {
      fCurrentManip->Attach(0);
   }

   switch (type) {
      case(kClipNone): {
         fCurrentClip = 0;
         break;
      }
      case(kClipPlane): {
         fCurrentClip = fClipPlane;
         break;
      }
      case(kClipBox): {
         fCurrentClip = fClipBox;
         break;
      }
      default: {
         Error("TGLScene::SetCurrentClip" , "Unknown clip type");
         break;
      }
   }

   // If editing clip, it is attached to manipulator
   if (edit) {
      fCurrentManip->Attach(fCurrentClip);

      // The clip object becomes out effective selection from users
      // perspective so clear the internal
      fSelectedPhysical = 0;
   }
}

//______________________________________________________________________________
void TGLScene::SetCurrentManip(EManipType type)
{
   switch (type) {
      case kManipTrans: {
         fTransManip.Attach(fCurrentManip->GetAttached());
         fCurrentManip = &fTransManip;
         break;
      }
      case kManipScale: {
         fScaleManip.Attach(fCurrentManip->GetAttached());
         fCurrentManip = &fScaleManip;
         break;
      }
      case kManipRotate: {
         fRotateManip.Attach(fCurrentManip->GetAttached());
         fCurrentManip = &fRotateManip;
         break;
      }
      default: {
         Error("TGLScene::SetCurrentManip", "invalid manipulator type");
         break;
      }
   }
}

//______________________________________________________________________________
const TGLBoundingBox & TGLScene::BoundingBox() const
{
   // Update (if required) and return the scene bounding box
   // Encapsulates all physical shapes bounding box with axes aligned box
   if (!fBoundingBoxValid) {
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
         TGLBoundingBox box = physicalShape->BoundingBox();
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
   return fBoundingBox;
}

//______________________________________________________________________________
void TGLScene::Dump() const
{
   // Output simple scene stats to std::cout
   std::cout << "Scene: " << fLogicalShapes.size() << " Logicals / " << fPhysicalShapes.size() << " Physicals " << std::endl;
}

//______________________________________________________________________________
UInt_t TGLScene::SizeOf() const
{
   // Return memory cost of scene
   // Warning: NOT CORRECT at present - doesn't correctly calculate size
   // of logical shapes with dynamic internal contents
   UInt_t size = sizeof(this);

   std::cout << "Size: Scene Only " << size << std::endl;

   LogicalShapeMapCIt_t logicalShapeIt = fLogicalShapes.begin();
   const TGLLogicalShape * logicalShape;
   while (logicalShapeIt != fLogicalShapes.end()) {
      logicalShape = logicalShapeIt->second;
      size += sizeof(*logicalShape);
      ++logicalShapeIt;
   }

   std::cout << "Size: Scene + Logical Shapes " << size << std::endl;

   PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physicalShape;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physicalShape = physicalShapeIt->second;
      size += sizeof(*physicalShape);
      ++physicalShapeIt;
   }

   std::cout << "Size: Scene + Logical Shapes + Physical Shapes " << size << std::endl;

   return size;
}

//______________________________________________________________________________
void TGLScene::ResetDrawStats(const TGLDrawFlags & flags)
{
   // Reset internal draw stats
   fDrawStats.fFlags = flags;
   fDrawStats.fOpaque = 0;
   fDrawStats.fTrans = 0;
   fDrawStats.fPixelLOD = 0;
   fDrawStats.fByShape.clear();
}

//______________________________________________________________________________
void TGLScene::UpdateDrawStats(const TGLPhysicalShape & shape, const TGLDrawFlags & flags)
{
   // Update draw stats, for newly drawn 'shape'

   // Update opaque/transparent draw count
   if (shape.IsTransparent()) {
      ++fDrawStats.fTrans;
   } else {
      ++fDrawStats.fOpaque;
   }

   if (flags.LOD() == TGLDrawFlags::kLODPixel) {
      fDrawStats.fPixelLOD++;
   }

   // By type only needed for debug currently
   if (gDebug>3) {
      // Update the stats
      std::string shapeType = shape.GetLogical().IsA()->GetName();
      typedef std::map<std::string, UInt_t>::iterator MapIt_t;
      MapIt_t statIt = fDrawStats.fByShape.find(shapeType);

      if (statIt == fDrawStats.fByShape.end()) {
         //do not need to check insert(.....).second, because statIt was stats.end() before
         statIt = fDrawStats.fByShape.insert(std::make_pair(shapeType, 0u)).first;
      }

      statIt->second++;
   }
}

//______________________________________________________________________________
void TGLScene::DumpDrawStats()
{
   // Output draw stats to std::cout

   // Draw counts
   if (gDebug>2) {
      std::string style;
      switch (fDrawStats.fFlags.Style()) {
         case TGLDrawFlags::kFill: {
            style = "Filled Polys";
            break;
         }
         case TGLDrawFlags::kWireFrame: {
            style = "Wireframe";
            break;
         }
         case TGLDrawFlags::kOutline: {
            style = "Outline";
            break;
         }
      }
      Info("TGLScene::DumpDrawStats()", "Drew scene (%s / %i LOD) - %i (Op %i Trans %i) %i pixel",
         style.c_str(),
         fDrawStats.fFlags.LOD(),
         fDrawStats.fOpaque + fDrawStats.fTrans,
         fDrawStats.fOpaque,
         fDrawStats.fTrans,
         fDrawStats.fPixelLOD);
   }

   // By shape type counts
   if (gDebug>3) {
      std::map<std::string, UInt_t>::const_iterator it = fDrawStats.fByShape.begin();
      while (it != fDrawStats.fByShape.end()) {
         std::cout << it->first << " (" << it->second << ")\t";
         it++;
      }
      std::cout << std::endl;
   }
}
