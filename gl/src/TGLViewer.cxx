// @(#)root/gl:$Name:  $:$Id: TGLViewer.cxx,v 1.10 2005/07/14 19:13:04 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header!!!

#include "TGLViewer.h"
#include "TGLIncludes.h"
#include "TGLStopwatch.h"
#include "TGLDisplayListCache.h"
#include "TError.h"


#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLStopwatch.h"
#include "TGLSceneObject.h" // For TGLFaceSet

#include "TColor.h"

ClassImp(TGLViewer)

//______________________________________________________________________________
TGLViewer::TGLViewer() :
   fPerspectiveCamera(),
   fOrthoXOYCamera(TGLOrthoCamera::kXOY),
   fOrthoYOZCamera(TGLOrthoCamera::kYOZ),
   fOrthoXOZCamera(TGLOrthoCamera::kXOZ),
   fCurrentCamera(&fPerspectiveCamera),
   fInternalRebuild(kFALSE), 
   fAcceptedAllPhysicals(kTRUE),
   fInternalPIDs(kFALSE), 
   fNextInternalPID(1), // 0 reserved
   fComposite(0), fCSLevel(0),
   fAcceptedPhysicals(0), 
   fRejectedPhysicals(0),
   fRedrawTimer(0),
   fNextSceneLOD(kHigh),
   fClipPlane(1.0, 0.0, 0.0, 0.0),
   fUseClipPlane(kFALSE),
   fDrawAxes(kFALSE),
   fInitGL(kFALSE),
   fDebugMode(kFALSE)
{
   fRedrawTimer = new TGLRedrawTimer(*this);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
}

//______________________________________________________________________________
Bool_t TGLViewer::PreferLocalFrame() const
{
   return kTRUE;
}

//______________________________________________________________________________
void TGLViewer::BeginScene()
{
   if (!fScene.TakeLock(TGLScene::kModifyLock)) {
      return;
   }

   UInt_t destroyedLogicals = 0;
   UInt_t destroyedPhysicals = 0;

   TGLStopwatch stopwatch;
   if (gDebug>2 || fDebugMode) {
      stopwatch.Start();
   }

   // External rebuild?
   if (!fInternalRebuild) 
   {
      // Potentially using external physical IDs
      fInternalPIDs = kFALSE;

      // Reset camera interest to ensure we respond to
      // new scene range
      CurrentCamera().ResetInterest();

      // External rebuilds could potentially invalidate all logical and
      // physical shapes - including any modified physicals
      // Physicals must be removed first
      destroyedPhysicals = fScene.DestroyPhysicals(kTRUE); // include modified
      destroyedLogicals = fScene.DestroyLogicals();

      // Purge out the DL cache - not required once shapes do this themselves properly
      TGLDisplayListCache::Instance().Purge();
   } else {
      // Internal rebuilds - destroy all non-modified physicals no longer of
      // interest to camera - retain logicals
      destroyedPhysicals = fScene.DestroyPhysicals(kFALSE, &CurrentCamera()); // excluded modified
   }

   // Reset internal physical ID counter
   fNextInternalPID = 1;
   
   // Potentially accepting all physicals from external client
   fAcceptedAllPhysicals = kTRUE;

  // Reset tracing info
   fAcceptedPhysicals = 0;
   fRejectedPhysicals = 0;

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::BeginScene", "destroyed %d physicals %d logicals in %f msec", 
            destroyedPhysicals, destroyedLogicals, stopwatch.End());
      fScene.Dump();
   }
}

//______________________________________________________________________________
void TGLViewer::EndScene()
{
   fScene.ReleaseLock(TGLScene::kModifyLock);

   // External scene build
   if (!fInternalRebuild) {
      // Setup camera unless scene is empty
      if (!fScene.BoundingBox().IsEmpty()) {
         SetupCameras(fScene.BoundingBox());
      }
      Invalidate();
   } else if (fInternalRebuild) {
      fInternalRebuild = kFALSE;
   }      

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::EndScene", "Added %d, rejected %d physicals, accepted all:%s", fAcceptedPhysicals, 
                                       fRejectedPhysicals, fAcceptedAllPhysicals ? "Yes":"No");
      fScene.Dump();
   }
}

//______________________________________________________________________________
Bool_t TGLViewer::RebuildScene()
{
   // If we accepted all offered physicals into the scene no point in 
   // rebuilding it
   if (fAcceptedAllPhysicals) {
      // For debug mode always force even if not required
      if (fDebugMode) {
         Info("TGLViewer::RebuildScene", "not required - all physicals previous accepted (FORCED anyway)");
      }
      else {
         if (gDebug>3) {
            Info("TGLViewer::RebuildScene", "not required - all physicals previous accepted");
         }
         return kFALSE;   
      }
   }
   // Update the camera interest (forced in debug mode) - if changed
   // scene should be rebuilt
   if (!CurrentCamera().UpdateInterest(fDebugMode)) {
      if (gDebug>3 || fDebugMode) {
         Info("TGLViewer::RebuildScene", "not required - no camera interest change");
      }
      return kFALSE;
   }
   
   // We are going to rebuild the scene - ensure any pending redraw timer cancelled now
   fRedrawTimer->Stop();

   if (gDebug>3 || fDebugMode) {
      Info("TGLViewer::RebuildScene", "required");
   }

   fInternalRebuild = kTRUE;
   
   TGLStopwatch timer;
   if (gDebug>2 || fDebugMode) {
      timer.Start();
   }

   // Request a scene fill
   FillScene();

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::RebuildScene", "rebuild complete in %f", timer.End());
   }

   // Need to invalidate/redraw via timer as under Win32 we are already inside the 
   // GUI(DoRedraw) thread - direct invalidation will be cleared when leaving
   fRedrawTimer->RequestDraw(20, kMed);

   return kTRUE;
}

//______________________________________________________________________________
Int_t TGLViewer::AddObject(const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using internal physical IDs

   // If this is called we are generating internal physical IDs
   fInternalPIDs = kTRUE;
   Int_t sections = AddObject(fNextInternalPID, buffer, addChildren);   
   return sections;
}

//______________________________________________________________________________
// TODO: Cleanup addChildren to UInt_t flag for full termination - how returned?
Int_t TGLViewer::AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using an external physical ID.

   // TODO: Break this up and make easier to understand. This is pretty convoluted
   // due to the large number of cases it has to deal with:
   // i) Exisiting physical and/or logical
   // ii) External provider can supply bounding box or not?
   // iii) Local/global reference frame
   // iv) Defered filling of some sections of the buffer
   // v) Internal or external physical IDs
   // vi) Composite components as special case
   //
   // The buffer filling means the function is re-entrant which adds to complication 

   if (physicalID == 0) {
      Error("TGLViewer::AddObject", "0 physical ID reserved");
      return TBuffer3D::kNone;
   }

   // Internal and external physical IDs cannot be mixed in a scene build
   if (fInternalPIDs && physicalID != fNextInternalPID) {
      Error("TGLViewer::AddObject", "invalid next physical ID - mix of internal + external IDs?");
      return TBuffer3D::kNone;
   }

   if (addChildren) {
      *addChildren = kFALSE;
   }
   
   // Scene should be modify locked
   if (fScene.CurrentLock() != TGLScene::kModifyLock) {
      Error("TGLViewer::AddObject", "expected scene to be in mofifed locked");
      // TODO: For the moment live with this - DrawOverlap() problems to discuss with Andrei
      // Just reject as pad will redraw anyway
      // assert(kFALSE);
      return TBuffer3D::kNone;
   }
   
   // Note that 'object' here is really a physical/logical pair described
   // in buffer + physical ID.

   // If adding component to a current partial composite do this now
   if (fComposite) {
      RootCsg::BaseMesh *newMesh = RootCsg::ConvertToMesh(buffer);
      // Solaris CC can't create stl pair with enumerate type
      fCSTokens.push_back(std::make_pair(static_cast<UInt_t>(TBuffer3D::kCSNoOp), newMesh));
      return TBuffer3D::kNone;
   }

   // TODO: Could be static and save possible double lookup?
   TGLLogicalShape * logical = fScene.FindLogical(reinterpret_cast<ULong_t>(buffer.fID));
   TGLPhysicalShape * physical = fScene.FindPhysical(physicalID);

   // Function can be called twice if extra buffer filling for logical 
   // is required - record last physical ID to detect
   static UInt_t lastPID = 0;

   // First attempt to add this physical 
   if (physicalID != lastPID) {
      // Existing physical
      if (physical) {
         assert(logical); // Have physical - should have logical
         
         if (addChildren) {
            // For internal PID we request all children even if we will reject them.
            // This ensures PID always represent same external entity.
            if (fInternalPIDs) {
               *addChildren = kTRUE;
            } else 
            // For external PIDs we check child interest as we may have reject children previously
            // with a different camera configuration
            {
               *addChildren = CurrentCamera().OfInterest(physical->BoundingBox());
            }
         }
         
         // Always increment the internal physical ID so they
         // match external object sequence
         if (fInternalPIDs) {
            fNextInternalPID++;
         }

         // We don't need anything more for this object
         return TBuffer3D::kNone; 
      }
      // New physical 
      else {
         // First test interest in camera - requires a bounding box
         TGLBoundingBox box;
         
         // If already have logical use it's BB
         if (logical) {
            box = logical->BoundingBox();
            //assert(!box.IsEmpty());
         }
         // else if bounding box in buffer valid use this
         else if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
            box.Set(buffer.fBBVertex);
            //assert(!box.IsEmpty());

         // otherwise we need to use raw points to build a bounding box with
         // If raw sections not set it will be requested by ValidateObjectBuffer
         // below and we will re-enter here
         } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
            box.SetAligned(buffer.NbPnts(), buffer.fPnts);
            //assert(!box.IsEmpty());
         }
      
         // Box is valid?
         if (!box.IsEmpty()) {
            // Test transformed box with camera
            box.Transform(TGLMatrix(buffer.fLocalMaster));
            Bool_t ofInterest = CurrentCamera().OfInterest(box);
            if (addChildren) {
               // For internal PID we request all children even if we will reject them.
               // This ensures PID always represent same external entity.
               if (fInternalPIDs) {
                  *addChildren = kTRUE;
               } else 
               // For external PID request children if physical of interest
               {
                  *addChildren = ofInterest;
               }
            }            
            // Physical is of interest?
            if (!ofInterest) {
               ++fRejectedPhysicals;
               fAcceptedAllPhysicals = kFALSE;

               // Always increment the internal physical ID so they
               // match external object sequence
               if (fInternalPIDs) {
                  fNextInternalPID++;
               }
               return TBuffer3D::kNone;
            } 
         }
      }

      // Need any extra sections in buffer?
      Int_t extraSections = ValidateObjectBuffer(buffer, 
                                                 logical == 0); // Need logical?
      if (extraSections != TBuffer3D::kNone) {         
         return extraSections;
      } else {
         lastPID = physicalID; // Will not to re-test interest
      }
   }

   if(lastPID != physicalID)
   {
      assert(kFALSE);
   }
   // By now we should need to add a physical at least
   if (physical) {
      assert(kFALSE);
      return TBuffer3D::kNone; 
   }

   // Create logical if required
   if (!logical) {
      assert(ValidateObjectBuffer(buffer,true) == TBuffer3D::kNone); // Buffer should be ready
      logical = CreateNewLogical(buffer);
      if (!logical) { 
         assert(kFALSE);
         return TBuffer3D::kNone;
      }
      // Add logical to scene
      fScene.AdoptLogical(*logical);
   }

   // Finally create the physical, binding it to the logical, and add to scene
   physical = CreateNewPhysical(physicalID, buffer, *logical);

   if (physical) { 
      fScene.AdoptPhysical(*physical);
      ++fAcceptedPhysicals;
      if (gDebug>3 && fAcceptedPhysicals%1000 == 0) {
         Info("TGLViewer::AddObject", "added %d physicals", fAcceptedPhysicals);
      }
   } else {
      assert(kFALSE);
   }

   // Always increment the internal physical ID so they
   // match external object sequence
   if (fInternalPIDs) {
      fNextInternalPID++;
   }

   // Reset last physical ID so can detect new one
   lastPID = 0;
   return TBuffer3D::kNone;
}

//______________________________________________________________________________
Bool_t TGLViewer::OpenComposite(const TBuffer3D & buffer, Bool_t * addChildren)
{
   assert(!fComposite);
   UInt_t extraSections = AddObject(buffer, addChildren);
   assert(extraSections == TBuffer3D::kNone);
   
   // If composite was created it is of interest - we want the rest of the
   // child components   
   if (fComposite) {
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
void TGLViewer::CloseComposite()
{
   // If we have a partially complete composite build it now
   if (fComposite) {
      // TODO: Why is this member and here - only used in BuildComposite()
      fCSLevel = 0;

      RootCsg::BaseMesh *resultMesh = BuildComposite();
      fComposite->SetFromMesh(resultMesh);
      delete resultMesh;
      for (UInt_t i = 0; i < fCSTokens.size(); ++i) delete fCSTokens[i].second;
      fCSTokens.clear();
      fComposite = 0;
   }
}

//______________________________________________________________________________
void TGLViewer::AddCompositeOp(UInt_t operation)
{
   fCSTokens.push_back(std::make_pair(operation, (RootCsg::BaseMesh *)0));
}

//______________________________________________________________________________
Int_t TGLViewer::ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t logical) const
{
   // kCore: Should always be filled
   if (!buffer.SectionsValid(TBuffer3D::kCore)) {
      assert(kFALSE);
      return TBuffer3D::kNone;
   }

   // Currently all physical parts (kBoundingBox / kShapeSpecific) of buffer are 
   // filled automatically if producer can - no need to ask 
   if (!logical) {
      return TBuffer3D::kNone;
   }

   // kRawSizes / kRaw: These are on demand based on shape type
   Bool_t needRaw = kFALSE;

   // We need raw tesselation in these cases:
   //
   // 1. Shape type is NOT kSphere / kTube / kTubeSeg / kCutTube / kComposite
   if (buffer.Type() != TBuffer3DTypes::kSphere  &&
       buffer.Type() != TBuffer3DTypes::kTube    &&
       buffer.Type() != TBuffer3DTypes::kTubeSeg &&
       buffer.Type() != TBuffer3DTypes::kCutTube && 
       buffer.Type() != TBuffer3DTypes::kComposite) {
      needRaw = kTRUE;
   }
   // 2. Sphere type is kSPHE, but the sphere is hollow and/or cut - we
   //    do not support native drawing of these currently
   else if (buffer.Type() == TBuffer3DTypes::kSphere) {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         if (!sphereBuffer->IsSolidUncut()) {
            needRaw = kTRUE;
         }
      } else {
         assert(kFALSE);
         return TBuffer3D::kNone;
      }
   }
   // 3. kBoundingBox is not filled - we generate a bounding box from 
   else if (!buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      needRaw = kTRUE;
   }
   // 3. kShapeSpecific is not filled - except in case of top level composite 
   else if (!buffer.SectionsValid(TBuffer3D::kShapeSpecific) && 
             buffer.Type() != TBuffer3DTypes::kComposite) {
      needRaw = kTRUE;
   }
   // 5. We are a component (not the top level) of a composite shape
   else if (fComposite) {
      needRaw = kTRUE;
   }

   if (needRaw && !buffer.SectionsValid(TBuffer3D::kRawSizes|TBuffer3D::kRaw)) {
      return TBuffer3D::kRawSizes|TBuffer3D::kRaw;
   } else {
      return TBuffer3D::kNone;
   }
}

//______________________________________________________________________________
TGLLogicalShape * TGLViewer::CreateNewLogical(const TBuffer3D & buffer) const
{
   // Buffer should now be correctly filled
   assert(ValidateObjectBuffer(buffer,true) == TBuffer3D::kNone);

   TGLLogicalShape * newLogical = 0;

   switch (buffer.Type()) {
   case TBuffer3DTypes::kLine:
      newLogical = new TGLPolyLine(buffer, buffer.fID);
      break;
   case TBuffer3DTypes::kMarker:
      newLogical = new TGLPolyMarker(buffer, buffer.fID);
      break;
   case TBuffer3DTypes::kSphere: {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         // We can only draw solid uncut spheres natively at present
         if (sphereBuffer->IsSolidUncut()) {
            newLogical = new TGLSphere(*sphereBuffer, sphereBuffer->fID);
         } else {
            newLogical = new TGLFaceSet(buffer, buffer.fID);
         }
      }
      else {
         assert(kFALSE);
      }
      break;
   }
   case TBuffer3DTypes::kTube:
   case TBuffer3DTypes::kTubeSeg:
   case TBuffer3DTypes::kCutTube: {
      const TBuffer3DTube * tubeBuffer = dynamic_cast<const TBuffer3DTube *>(&buffer);
      if (tubeBuffer)
      {
         newLogical = new TGLCylinder(*tubeBuffer, tubeBuffer->fID);
      }
      else {
         assert(kFALSE);
      }
      break;
   }
   case TBuffer3DTypes::kComposite: {
      // Create empty faceset and record partial complete composite object
      // Will be populated with mesh in CloseComposite()
      assert(!fComposite);
      fComposite = new TGLFaceSet(buffer, buffer.fID);
      newLogical = fComposite;
      break;
   }
   default:
      newLogical = new TGLFaceSet(buffer, buffer.fID);
      break;
   }

   return newLogical;
}

//______________________________________________________________________________
TGLPhysicalShape * TGLViewer::CreateNewPhysical(UInt_t ID, 
                                                    const TBuffer3D & buffer, 
                                                    const TGLLogicalShape & logical) const
{
   // Extract indexed color from buffer
   // TODO: Still required? Better use proper color triplet in buffer?
   Int_t colorIndex = buffer.fColor;
   if (colorIndex <= 1) colorIndex = 42; //temporary
   Float_t rgba[4] = { 0.0 };
   TColor *rcol = gROOT->GetColor(colorIndex);

   if (rcol) {
      rcol->GetRGB(rgba[0], rgba[1], rgba[2]);
   }
   
   // Extract transparency component - convert to opacity (alpha)
   rgba[3] = 1.f - buffer.fTransparency / 100.f;

   TGLPhysicalShape * newPhysical = new TGLPhysicalShape(ID, logical, buffer.fLocalMaster, 
                                                         buffer.fReflection, rgba);
   return newPhysical;
}

//______________________________________________________________________________
RootCsg::BaseMesh *TGLViewer::BuildComposite()
{
   const CSPart_t &currToken = fCSTokens[fCSLevel];
   UInt_t opCode = currToken.first;

   if (opCode != TBuffer3D::kCSNoOp) {
      ++fCSLevel;
      RootCsg::BaseMesh *left = BuildComposite();
      RootCsg::BaseMesh *right = BuildComposite();
      //RootCsg::BaseMesh *result = 0;
      switch (opCode) {
      case TBuffer3D::kCSUnion:
         return RootCsg::BuildUnion(left, right);
      case TBuffer3D::kCSIntersection:
         return RootCsg::BuildIntersection(left, right);
      case TBuffer3D::kCSDifference:
         return RootCsg::BuildDifference(left, right);
      default:
         Error("BuildComposite", "Wrong operation code %d\n", opCode);
         return 0;
      }
   } else return fCSTokens[fCSLevel++].second;
}

//______________________________________________________________________________
void TGLViewer::Draw()
{
   // Draw out the the current viewer/scene

   // Locking mainly for Win32 mutli thread safety - but no harm in all using it
   // During normal draws a draw lock is taken in other thread (Win32) in TViewerOpenGL
   // to ensure thread safety. For PrintObjects repeated Draw() calls are made.
   // If no draw lock taken get one now
   if (fScene.CurrentLock() != TGLScene::kDrawLock) {
      if (!fScene.TakeLock(TGLScene::kDrawLock)) {
         Error("TGLViewer::Draw", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
   }

   TGLStopwatch timer;
   UInt_t drawn = 0;
   if (gDebug>2) {
      timer.Start();
   }

   PreDraw();

   // Apply current camera projection (always as scene may be empty now but rebuilt
   // in which case camera must have been applied)
   fCurrentCamera->Apply(fScene.BoundingBox());

   // Something to draw?
   if (!fScene.BoundingBox().IsEmpty()) {
      // Draw axes. Still get's clipped - need to find a way to disable clips
      // for this
      if (fDrawAxes) {
         fScene.DrawAxes();
      }

      // Apply any clipping plane
      if (fUseClipPlane) {
         glEnable(GL_CLIP_PLANE0);
         glClipPlane(GL_CLIP_PLANE0, fClipPlane.CArr());
      } else {
         glDisable(GL_CLIP_PLANE0);
      }

      if (fNextSceneLOD == kHigh) {
         // High quality (final pass) draws have unlimited time to complete
         drawn = fScene.Draw(*fCurrentCamera, fNextSceneLOD);
      } else {
         // Other (interactive) draws terminate after 100 msec
         drawn = fScene.Draw(*fCurrentCamera, fNextSceneLOD, 100.0);
      }

      // Debug mode - draw some extra boxes
      if (fDebugMode) {
         glDisable(GL_LIGHTING);
         CurrentCamera().DrawDebugAids();

         // Green scene bounding box
         glColor3d(0.0, 1.0, 0.0);
         fScene.BoundingBox().Draw();
         glEnable(GL_LIGHTING);
       }
   }

   PostDraw();

   if (gDebug>2) {
      Info("TGLViewer::Draw()", "Drew %i at %i LOD in %f msec", drawn, fNextSceneLOD, timer.End());
      if (gDebug>3) {
         TGLDisplayListCache::Instance().Dump();
      }
   }

   // Release draw lock on scene
   fScene.ReleaseLock(TGLScene::kDrawLock);

   Bool_t redrawReq = kFALSE;

   // Debug mode have forced rebuilds only
   if (!fDebugMode) {
      // Final draw pass
      if (fNextSceneLOD == kHigh) {
         RebuildScene();
      } else {
         // Final draw pass required
         redrawReq = kTRUE;
      }
   } else {
      // Final draw pass required?
      redrawReq = fNextSceneLOD != kHigh;
   }

   // Request final pass high quality redraw via timer
   if (redrawReq) {
      fRedrawTimer->RequestDraw(100, kHigh);
   }
}

//______________________________________________________________________________
void TGLViewer::PreDraw()
{
   // GL work which must be done before each draw of scene
   MakeCurrent();

   // Initialise GL if not done
   if (!fInitGL) {
      InitGL();
   }

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   TGLUtil::CheckError();
}

//______________________________________________________________________________
void TGLViewer::PostDraw()
{
   // GL work which must be done after each draw of scene
   SwapBuffers();

   // Flush everything in case picking starts
   glFlush();

   TGLUtil::CheckError();
}

//______________________________________________________________________________
void TGLViewer::Invalidate(UInt_t redrawLOD)
{
   fNextSceneLOD = redrawLOD;
   fRedrawTimer->Stop();
}

//______________________________________________________________________________
Bool_t TGLViewer::Select(const TGLRect & rect)
{
   // Select lock should already been taken in other thread in 
   // TViewerOpenGL::DoSelect()
   if (fScene.CurrentLock() != TGLScene::kSelectLock) {
      Error("TGLViewer::Draw", "expected kSelectLock, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return kFALSE;
   }

   TGLRect glRect(rect);
   WindowToGL(glRect);
   fCurrentCamera->Apply(fScene.BoundingBox(), &glRect);

   MakeCurrent();
   Bool_t changed = fScene.Select(*fCurrentCamera);

   // Release select lock on scene before invalidation
   fScene.ReleaseLock(TGLScene::kSelectLock);

   if (changed) {
      Invalidate(kHigh);
   }

   return changed;
}

//______________________________________________________________________________
void TGLViewer::SetViewport(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   if (fScene.IsLocked()) {
      Error("TGLViewer::SetViewport", "expected kUnlocked, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return;
   }
   fViewport.Set(x, y, width, height);
   fCurrentCamera->SetViewport(fViewport);
   Invalidate();
}

//______________________________________________________________________________
void TGLViewer::SetCurrentCamera(ECamera camera)
{
   if (fScene.IsLocked()) {
      Error("TGLViewer::SetCurrentCamera", "expected kUnlocked, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return;
   }

   switch(camera) {
      case(kPerspective): {
         fCurrentCamera = &fPerspectiveCamera;
         break;
      }
      case(kXOY): {
         fCurrentCamera = &fOrthoXOYCamera;
         break;
      }
      case(kYOZ): {
         fCurrentCamera = &fOrthoYOZCamera;
         break;
      }
      case(kXOZ): {
         fCurrentCamera = &fOrthoXOZCamera;
         break;
      }
      default: {
         assert(kFALSE);
         break;
      }
   }

   // Ensure any viewport has been propigated to the current camera
   fCurrentCamera->SetViewport(fViewport);
}

//______________________________________________________________________________
void TGLViewer::SetupCameras(const TGLBoundingBox & box)
{
   if (fScene.IsLocked()) {
      Error("TGLViewer::SetupCameras", "expected kUnlocked, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return;
   }

   fPerspectiveCamera.Setup(box);
   fOrthoXOYCamera.Setup(box);
   fOrthoYOZCamera.Setup(box);
   fOrthoXOZCamera.Setup(box);
}
