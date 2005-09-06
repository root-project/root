// @(#)root/gl:$Name:  $:$Id: TGLViewer.cxx,v 1.15 2005/09/05 11:03:27 brun Exp $
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

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLStopwatch.h"
#include "TGLSceneObject.h" // For TGLFaceSet

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TVirtualPad.h" // Remove when pad removed - use signal

#include "TColor.h"
#include "TError.h"

// For event type translation ExecuteEvent
#include "Buttons.h"
#include "GuiTypes.h"

// Remove - replace with TGLManager
#include "TVirtualGL.h"
#include "TGLRenderArea.h"

#include "KeySymbols.h"
#include "TContextMenu.h"

ClassImp(TGLViewer)

//______________________________________________________________________________
TGLViewer::TGLViewer(TVirtualPad * pad, Int_t x, Int_t y, 
                     UInt_t width, UInt_t height) :
   fPad(pad),
   fContextMenu(0),
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
   fAction(kNone), fStartPos(0,0), fLastPos(0,0), fActiveButtonID(0),
   fDrawStyle(kFill),
   fRedrawTimer(0),
   fNextSceneLOD(kHigh),
   fLightState(kLightMask), // All on
   fClipPlane(1.0, 0.0, 0.0, 0.0),
   fUseClipPlane(kFALSE),
   fDrawAxes(kFALSE),
   fInitGL(kFALSE),
   fDebugMode(kFALSE),
   fAcceptedPhysicals(0), 
   fRejectedPhysicals(0),
   fIsPrinting(kFALSE),
   fGLWindow(0)
{
   fRedrawTimer = new TGLRedrawTimer(*this);
   SetViewport(x, y, width, height);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
   delete fContextMenu;
   delete fRedrawTimer;
   fPad->ReleaseViewer3D();   
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
      RequestDraw();
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
   // TODO: Just marking modified doesn't seem to result in pad repaint - need to check on
   //fPad->Modified();
   fPad->Paint();

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
void TGLViewer::InitGL
()
{
   assert(!fInitGL);

   // GL initialisation 
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_BLEND);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glClearColor(0.0, 0.0, 0.0, 0.0);
   glClearDepth(1.0);

   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodelAmb);
   
   TGLUtil::CheckError();
   fInitGL = kTRUE;
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

//______________________________________________________________________________
void TGLViewer::SetupLights()
{
   // Setup lights

   // Locate static light source positions - this is done once only
   // after the scene has been populated 
   if (!fScene.BoundingBox().IsEmpty()) {
      // Find camera offset to scene bounding box so lights can be
      // arranged round it

      // Apply camera so can extract the eye point
      fCurrentCamera->Apply(fScene.BoundingBox());
      TGLBoundingBox box = fScene.BoundingBox();
      TGLVector3 lightVector = fCurrentCamera->EyePoint() - box.Center();

      // Reset the modelview to lights are placed in fixed eye space
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      // Calculate a light Z distance - to center of box in eye coords
      // Pull forward slightly (0.85) to avoid to sharp a cutoff
      Double_t lightZ = lightVector.Mag() * 0.85;

      // Calculate a sphere radius to arrange lights round
      Double_t lightRadius = box.Extents().Mag() * 3.0;

      // 0: Front
      // 1: Top   
      // 2: Bottom
      // 3: Left
      // 4: Right
      Float_t pos0[] = {     0.0,              0.0,     0.0, 1.0};
      Float_t pos1[] = {     0.0,      lightRadius, -lightZ, 1.0};
      Float_t pos2[] = {     0.0,     -lightRadius, -lightZ, 1.0};
      Float_t pos3[] = {-lightRadius,          0.0, -lightZ, 1.0};
      Float_t pos4[] = { lightRadius,          0.0, -lightZ, 1.0};

      Float_t frontLightColor[] = {0.35, 0.35, 0.35, 1.0};
      Float_t sideLightColor[] = {0.7, 0.7, 0.7, 1.0};
      glLightfv(GL_LIGHT0, GL_POSITION, pos0);
      glLightfv(GL_LIGHT0, GL_DIFFUSE, frontLightColor);
      glLightfv(GL_LIGHT1, GL_POSITION, pos1);
      glLightfv(GL_LIGHT1, GL_DIFFUSE, sideLightColor);
      glLightfv(GL_LIGHT2, GL_POSITION, pos2);
      glLightfv(GL_LIGHT2, GL_DIFFUSE, sideLightColor);
      glLightfv(GL_LIGHT3, GL_POSITION, pos3);
      glLightfv(GL_LIGHT3, GL_DIFFUSE, sideLightColor);
      glLightfv(GL_LIGHT4, GL_POSITION, pos4);
      glLightfv(GL_LIGHT4, GL_DIFFUSE, sideLightColor);
   }

   // TODO: Could detect state change and only adjust if a change
   for (UInt_t light = 0; (1<<light) < kLightMask; light++) {
      if ((1<<light) & fLightState) {
         glEnable(GLenum(GL_LIGHT0 + light));
      } else {
         glDisable(GLenum(GL_LIGHT0 + light));
      }
   }
}

//______________________________________________________________________________
void TGLViewer::DoDraw()
{
   // Draw out the the current viewer/scene

   // Locking mainly for Win32 mutli thread safety - but no harm in all using it
   // During normal draws a draw lock is taken in other thread (Win32) in RequestDraw()
   // to ensure thread safety. For PrintObjects repeated Draw() calls are made.
   // If no draw lock taken get one now
   if (fScene.CurrentLock() != TGLScene::kDrawLock) {
      if (!fScene.TakeLock(TGLScene::kDrawLock)) {
         Error("TGLViewer::DoDraw", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
   }

   fRedrawTimer->Stop();

   TGLStopwatch timer;
   UInt_t drawn = 0;
   if (gDebug>2) {
      timer.Start();
   }

   // GL pre draw setup
   if (!fIsPrinting) PreDraw();

   SetupLights();

   // Apply current camera projection - always do this even if scene is empty and we don't draw, 
   // as scene will likely be rebuilt, requiring camera interest and caching needs to be established
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
         drawn = fScene.Draw(*fCurrentCamera, fDrawStyle, fNextSceneLOD);
      } else {
         // Other (interactive) draws terminate after 100 msec
         drawn = fScene.Draw(*fCurrentCamera, fDrawStyle, fNextSceneLOD, 100.0);
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
      Info("TGLViewer::DoDraw()", "Took %f msec", timer.End());
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

   // Setup GL for current draw style - fill, wireframe, outline
   // Any GL modifications need to be defered until drawing time - 
   // to ensure we are in correct thread/context under Windows
   // TODO: Could detect change and only mod if changed for speed
   switch (fDrawStyle) {
      case (kFill): {
         glEnable(GL_LIGHTING);
         glEnable(GL_CULL_FACE);
         glPolygonMode(GL_FRONT, GL_FILL);
         glClearColor(0.0, 0.0, 0.0, 1.0); // Black
         break;
      }
      case (kWireFrame): {
         glDisable(GL_CULL_FACE);
         glDisable(GL_LIGHTING);
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glClearColor(0.0, 0.0, 0.0, 1.0); // Black
         break;
      }
      case (kOutline): {
         glEnable(GL_LIGHTING);
         glEnable(GL_CULL_FACE);
         glPolygonMode(GL_FRONT, GL_FILL);
         glClearColor(1.0, 1.0, 1.0, 1.0); // White
         break;
      }
      default: {
         assert(kFALSE);
      }
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
void TGLViewer::MakeCurrent() const
{
   fGLWindow->MakeCurrent();
}

//______________________________________________________________________________
void TGLViewer::SwapBuffers() const
{
   if (fScene.CurrentLock() != TGLScene::kDrawLock && 
      fScene.CurrentLock() != TGLScene::kSelectLock) {
      Error("TGLViewer::MakeCurrent", "scene is %s", TGLScene::LockName(fScene.CurrentLock()));   
   }
   fGLWindow->SwapBuffers();
}

//______________________________________________________________________________
void TGLViewer::RequestDraw(UInt_t LOD)
{
   fNextSceneLOD = LOD;
   fRedrawTimer->Stop();
   
   // Take scene lock - to be revisited
   if (!fScene.TakeLock(TGLScene::kDrawLock)) {
      // If taking drawlock fails the previous draw is still in progress
      // set timer to do this one later
      if (gDebug>3) {
         Info("TGLViewer::DoRedraw", "scene drawlocked - requesting another draw");
      }
      fRedrawTimer->RequestDraw(100, fNextSceneLOD);
      return;
   }
   
   gVirtualGL->DrawViewer(this);
}

//______________________________________________________________________________
Bool_t TGLViewer::DoSelect(const TGLRect & rect)
{
   // Select lock should already been taken in other thread in 
   // TGLViewer::DoSelect()
   if (fScene.CurrentLock() != TGLScene::kSelectLock) {
      Error("TGLViewer::Draw", "expected kSelectLock, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return kFALSE;
   }

   MakeCurrent();

   TGLRect glRect(rect);
   WindowToGL(glRect);
   fCurrentCamera->Apply(fScene.BoundingBox(), &glRect);

   Bool_t changed = fScene.Select(*fCurrentCamera, fDrawStyle);

   // Release select lock on scene before invalidation
   fScene.ReleaseLock(TGLScene::kSelectLock);

   if (changed) {
      RequestDraw(kHigh);

      // Inform external client selection has been modified
      SelectionChanged();
   }

   return changed;
}

//______________________________________________________________________________
void TGLViewer::RequestSelect(UInt_t x, UInt_t y)
{
   // Take select lock on scene immediately we enter here - it is released
   // in the other (drawing) thread - see TGLViewer::Select()
   // Removed when gVirtualGL removed
   if (!fScene.TakeLock(TGLScene::kSelectLock)) {
      return;
   }

   // TODO: Check only the GUI thread ever enters here & DoSelect.
   // Then TVirtualGL and TGLKernel can be obsoleted.
   TGLRect selectRect(x, y, 3, 3); // TODO: Constant somewhere
   gVirtualGL->SelectViewer(this, &selectRect); 
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

   // Can't do this until gVirtualGL has been setup - change with TGLManager
   //RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::SetCurrentCamera(ECamera camera)
{
   if (fScene.IsLocked()) {
      Error("TGLViewer::SetCurrentCamera", "expected kUnlocked, found %s", TGLScene::LockName(fScene.CurrentLock()));
      return;
   }

   switch(camera) {
      case(kCameraPerspective): {
         fCurrentCamera = &fPerspectiveCamera;
         break;
      }
      case(kCameraXOY): {
         fCurrentCamera = &fOrthoXOYCamera;
         break;
      }
      case(kCameraYOZ): {
         fCurrentCamera = &fOrthoYOZCamera;
         break;
      }
      case(kCameraXOZ): {
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

   // And viewer is redrawn
   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::ToggleLight(ELight light)
{
   // Toggle supplied light on/off

   // N.B. We can't directly call glEnable here as may not be in correct gl context
   // adjust mask and set when drawing
   if (light >= kLightMask) {
      assert(kFALSE);
      return;
   }

   fLightState ^= light;

   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::ToggleAxes()
{
   fDrawAxes = !fDrawAxes;
   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::ToggleClip()
{
   fUseClipPlane = !fUseClipPlane;
   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::SetClipPlaneEq(const TGLPlane & eqn)
{
   fClipPlane.Set(eqn);
   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::SetSelectedColor(const Float_t rgba[4])
{
   if (fScene.SetSelectedColor(rgba)) {
      RequestDraw();
   }
}

//______________________________________________________________________________
void TGLViewer::SetColorOnSelectedFamily(const Float_t rgba[4])
{
   if (fScene.SetColorOnSelectedFamily(rgba)) {
      RequestDraw();
   }
}

//______________________________________________________________________________
void TGLViewer::SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale)
{
   if (fScene.SetSelectedGeom(trans, scale)) {
      RequestDraw();
   }
}

//______________________________________________________________________________
void TGLViewer::SelectionChanged() 
{ 
   Emit("SelectionChanged()"); 
}

//______________________________________________________________________________
Int_t TGLViewer::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
   // Can't track the indvidual objects in rollover. Just set the viewer as the
   // selected object, and return 0 (object identified) so we receive ExecuteEvent calls
   gPad->SetSelected(this);
   return 0;
}

//______________________________________________________________________________
void TGLViewer::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   /*enum EEventType {
   kNoEvent       =  0,
   kButton1Down   =  1, kButton2Down   =  2, kButton3Down   =  3, kKeyDown  =  4,
   kButton1Up     = 11, kButton2Up     = 12, kButton3Up     = 13, kKeyUp    = 14,
   kButton1Motion = 21, kButton2Motion = 22, kButton3Motion = 23, kKeyPress = 24,
   kButton1Locate = 41, kButton2Locate = 42, kButton3Locate = 43,
   kMouseMotion   = 51, kMouseEnter    = 52, kMouseLeave    = 53,
   kButton1Double = 61, kButton2Double = 62, kButton3Double = 63
   
   enum EGEventType {
   kGKeyPress, kKeyRelease, kButtonPress, kButtonRelease,
   kMotionNotify, kEnterNotify, kLeaveNotify, kFocusIn, kFocusOut,
   kExpose, kConfigureNotify, kMapNotify, kUnmapNotify, kDestroyNotify,
   kClientMessage, kSelectionClear, kSelectionRequest, kSelectionNotify,
   kColormapNotify, kButtonDoubleClick, kOtherEvent*/

   // Map our event EEventType (base/inc/Buttons.h) back to Event_t (base/inc/GuiTypes.h)
   // structure, and call appropriate HandleXXX() function 
   Event_t eventSt;
   eventSt.fX = px;
   eventSt.fY = py;

   switch (event) {
      case kButton1Down:
      case kButton1Up:
      {
         eventSt.fCode = kButton1;
         eventSt.fType = kButton1Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      case kButton2Down:
      case kButton2Up:
      {
         eventSt.fCode = kButton2;
         eventSt.fType = kButton2Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      case kButton3Down:
      case kButton3Up:
      {
         eventSt.fCode = kButton3;
         eventSt.fType = kButton3Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      case kButton1Double:
      case kButton2Double:
      case kButton3Double:
      {
         eventSt.fCode = kButton1Double ? kButton1 : kButton2Double ? kButton2 : kButton3;
         eventSt.fType = kButtonDoubleClick;
         HandleDoubleClick(&eventSt);
      }
      case kButton1Motion:
      case kButton2Motion:
      case kButton3Motion:
      {
         eventSt.fCode = kButton1Motion ? kButton1 : kButton2Motion ? kButton2 : kButton3;
         eventSt.fType = kMotionNotify;
         HandleMotion(&eventSt);
      }
      case kKeyPress: // We only care about full key 'presses' not individual down/up
      {
         eventSt.fType = kKeyRelease;
         eventSt.fCode = px; // px contains key code - need modifiers from somewhere
         HandleKey(&eventSt);
      }
      default: 
      {
         assert(kFALSE);
      }
   }
};

//______________________________________________________________________________
Bool_t TGLViewer::HandleEvent(Event_t *event)
{
   if (event->fType == kFocusIn) {
      assert(fAction == kNone);
      fAction = kNone;
   }
   if (event->fType == kFocusOut) {
      fAction = kNone;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleButton(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>2) {
         Info("TGLViewer::HandleButton", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   // Only process one action/button down/up pairing - block others
   if (fAction != kNone) {
      if (event->fType == kButtonPress ||
          (event->fType == kButtonRelease && event->fCode != fActiveButtonID)) {
         return kFALSE;
      }
   }
   
   // Button DOWN
   if (event->fType == kButtonPress) {
      Bool_t grabPointer = kFALSE;

      // Record active button for release
      fActiveButtonID = event->fCode;

      // Record mouse start
      fStartPos.fX = fLastPos.fX = event->fX;
      fStartPos.fY = fLastPos.fY = event->fY;
      
      switch(event->fCode) {
         // LEFT mouse button
         case(kButton1): {
            if (event->fState & kKeyShiftMask) {
               RequestSelect(event->fX, event->fY);

               // TODO: If no selection start a box select
            } else {
               fAction = kRotate;
               grabPointer = kTRUE;
            }
            break;
         }
         // MID mouse button
         case(kButton2): {
            if (event->fState & kKeyShiftMask) {
               RequestSelect(event->fX, event->fY);
               // Start object drag
               if (fScene.GetSelected()) {
                  fAction = kDrag;
                  grabPointer = kTRUE;
               }
            } else {
               fAction = kTruck;
               grabPointer = kTRUE;
            }
            break;
         }
         // RIGHT mouse button
         case(kButton3): {
            // Shift + Right mouse - select+context menu
            if (event->fState & kKeyShiftMask) {
               RequestSelect(event->fX, event->fY);
               const TGLPhysicalShape * selected = fScene.GetSelected();
               if (selected) {
                  if (!fContextMenu) {
                     fContextMenu = new TContextMenu("glcm", "GL Viewer Context Menu");
                  }
                  selected->InvokeContextMenu(*fContextMenu, event->fX, event->fY);
               }
            } else {
               fAction = kDolly;
               grabPointer = kTRUE;
            }
            break;
         }
      }
   }
   // Button UP
   else if (event->fType == kButtonRelease) {
      // TODO: Check on Linux - on Win32 only see button release events
      // for mouse wheel
      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         case(kButton4): {
            // Zoom out (adjust camera FOV)
            if (CurrentCamera().Zoom(-30, event->fState & kKeyControlMask, 
                                          event->fState & kKeyShiftMask)) { //TODO : val static const somewhere
               RequestDraw();
            }
            break;
         }
         case(kButton5): {
            // Zoom in (adjust camera FOV)
            if (CurrentCamera().Zoom(+30, event->fState & kKeyControlMask, 
                                          event->fState & kKeyShiftMask)) { //TODO : val static const somewhere
               RequestDraw();
            }
            break;
         }
      }
      fAction = kNone;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleDoubleClick(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleDoubleClick", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   // Reset interactive camera mode on button double
   // click (unless mouse wheel)
   if (event->fCode != kButton4 && event->fCode != kButton5) {
      CurrentCamera().Reset();
      fStartPos.fX = fLastPos.fX = event->fX;
      fStartPos.fY = fLastPos.fY = event->fY;
      RequestDraw();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleConfigureNotify(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleConfigure", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   if (event) {
      SetViewport(event->fX, event->fY, event->fWidth, event->fHeight);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleKey(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleKey", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   char tmp[10] = {0};
   UInt_t keysym = 0;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   
   Bool_t redraw = kFALSE;

   switch (keysym) {
   case kKey_Plus:
   case kKey_J:
   case kKey_j:
      redraw = CurrentCamera().Dolly(10, event->fState & kKeyControlMask, 
                                         event->fState & kKeyShiftMask); //TODO : val static const somewhere
      break;
   case kKey_Minus:
   case kKey_K:
   case kKey_k:
      redraw = CurrentCamera().Dolly(-10, event->fState & kKeyControlMask, 
                                          event->fState & kKeyShiftMask); //TODO : val static const somewhere
      break;
   case kKey_R:
   case kKey_r:
      fDrawStyle = kFill;
      redraw = kTRUE;
      break;
   case kKey_W:
   case kKey_w:
      fDrawStyle = kWireFrame;
      redraw = kTRUE;
      break;
   case kKey_T:
   case kKey_t:
      fDrawStyle = kOutline;
      redraw = kTRUE;
      break;
   case kKey_Up:
      redraw = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 0, 5);
      break;
   case kKey_Down:
      redraw = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 0, -5);
      break;
   case kKey_Left:
      redraw = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), -5, 0);
      break;
   case kKey_Right:
      redraw = CurrentCamera().Truck(fViewport.CenterX(), fViewport.CenterY(), 5, 0);
      break;
   // Toggle debugging mode
   case kKey_D:
   case kKey_d:
      fDebugMode = !fDebugMode;
      redraw = kTRUE;
      Info("OpenGL viewer debug mode : ", fDebugMode ? "ON" : "OFF");
      break;
   // Forced rebuild for debugging mode
   case kKey_Space:
      if (fDebugMode) {
         Info("OpenGL viewer FORCED rebuild", "");
         RebuildScene();
      }
   }

   if (redraw) {
      RequestDraw();
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleMotion(Event_t *event)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleMotion", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   if (!event) {
      return kFALSE;
   }
   
   Bool_t redraw = kFALSE;
   
   Int_t xDelta = event->fX - fLastPos.fX;
   Int_t yDelta = event->fY - fLastPos.fY;
   
   // Camera interface requires GL coords - Y inverted
   if (fAction == kRotate) {
      redraw = CurrentCamera().Rotate(xDelta, -yDelta);
   } else if (fAction == kTruck) {
      redraw = CurrentCamera().Truck(event->fX, fViewport.Y() - event->fY, xDelta, -yDelta);
   } else if (fAction == kDolly) {
      redraw = CurrentCamera().Dolly(xDelta, event->fState & kKeyControlMask, 
                                                 event->fState & kKeyShiftMask);
   } else if (fAction == kDrag) {
      const TGLPhysicalShape * selected = fScene.GetSelected();
      if (selected) {
         TGLVector3 shift = CurrentCamera().ProjectedShift(selected->BoundingBox().Center(), xDelta, -yDelta);

         // Don't modify selected directly as scene needs to invalidate bounding box
         // hence will only give us a const handle on selected
         redraw = fScene.ShiftSelected(shift);

         // Inform external client selection has been modified
         SelectionChanged();
      }
   }

   fLastPos.fX = event->fX;
   fLastPos.fY = event->fY;
   
   if (redraw) {
      RequestDraw();
   }
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleExpose(Event_t *)
{
   if (fScene.IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleExpose", "ignored - scene is %s", TGLScene::LockName(fScene.CurrentLock()));
      }
      return kFALSE;
   }

   RequestDraw(kHigh);
   return kTRUE;
}

