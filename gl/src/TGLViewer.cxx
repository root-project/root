// @(#)root/gl:$Name:  $:$Id: TGLViewer.cxx,v 1.68 2007/06/23 21:23:22 brun Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLViewer.h"
#include "TGLIncludes.h"
#include "TGLStopwatch.h"
#include "TGLRnrCtx.h"
#include "TGLSelectBuffer.h"
#include "TGLLightSet.h"
#include "TGLClip.h"
#include "TGLManipSet.h"

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLObject.h"
#include "TGLStopwatch.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGLFaceSet.h"
#include "TGLPolyLine.h"
#include "TGLPolyMarker.h"
#include "TGLCylinder.h"
#include "TGLSphere.h"
#include "TGLOutput.h"

#include "TVirtualPad.h" // Remove when pad removed - use signal
#include "TAtt3D.h"      // Remove when PadPaint delegated to PadScene.
#include "TVirtualX.h"

#include "TH2.h"         // Preliminary support for GL plot painters
#include "TH2GL.h"
#include "TF2.h"
#include "TF2GL.h"

#include "TMath.h"
#include "TColor.h"
#include "TError.h"
#include "TClass.h"
#include "TROOT.h"

// For event type translation ExecuteEvent
#include "Buttons.h"
#include "GuiTypes.h"

// Remove - replace with TGLManager
#include "TVirtualGL.h"
//#include "TGLRenderArea.h"
#include "TGLWidget.h"
#include "TGLViewerEditor.h"

#include "KeySymbols.h"
#include "TContextMenu.h"

#include <TBaseClass.h>

//______________________________________________________________________
// TGLViewer
//
// Base GL viewer object - used by both standalone and embedded (in pad)
// GL. Contains core viewer objects :
//
// GL scene (fScene) - collection of main drawn objects - see TGLStdScene
// Cameras (fXXXXCamera) - ortho and perspective cameras - see TGLCamera
// Clipping (fClipXXXX) - collection of clip objects - see TGLClip
// Manipulators (fXXXXManip) - collection of manipulators - see TGLManip
//
// It maintains the current active draw styles, clipping object,
// manipulator, camera etc.
//
// TGLViewer is 'GUI free' in that it does not derive from any ROOT GUI
// TGFrame etc - see TGLSAViewer for this. However it contains GUI
// GUI style methods HandleButton() etc to which GUI events can be
// directed from standalone frame or embedding pad to perform
// interaction.
//
// For embedded (pad) GL this viewer is created directly by plugin
// manager. For standalone the derived TGLSAViewer is.
//

ClassImp(TGLViewer)

//______________________________________________________________________________
TGLViewer::TGLViewer(TVirtualPad * pad, Int_t x, Int_t y,
                     Int_t width, Int_t height) :
   fPad(pad),
   fContextMenu(0),
   fPerspectiveCameraXOZ(TGLVector3(1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // XOZ floor
   fPerspectiveCameraYOZ(TGLVector3(0.0, 1.0, 0.0), TGLVector3(1.0, 0.0, 0.0)), // YOZ floor
   fPerspectiveCameraXOY(TGLVector3(1.0, 0.0, 0.0), TGLVector3(0.0, 0.0,-1.0)), // XOY floor
   fOrthoXOYCamera(TGLOrthoCamera::kXOY),
   fOrthoXOZCamera(TGLOrthoCamera::kXOZ),
   fOrthoZOYCamera(TGLOrthoCamera::kZOY),
   fCurrentCamera(&fPerspectiveCameraXOZ),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fInternalRebuild(kFALSE),
   fPostSceneBuildSetup(kFALSE),
   fAcceptedAllPhysicals(kTRUE),
   fForceAcceptAll(kFALSE),
   fInternalPIDs(kFALSE),
   fNextInternalPID(1), // 0 reserved
   fComposite(0), fCSLevel(0),
   fAction(kDragNone), fLastPos(0,0), fActiveButtonID(0),
   fRedrawTimer(0),
   fClearColor(1),
   fAxesType(TGLUtil::kAxesNone),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fInitGL(kFALSE),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fAcceptedPhysicals(0),
   fRejectedPhysicals(0),
   fIsPrinting(kFALSE),
   fGLWindow(0),
   fGLDevice(-1),
   fGLCtxId(0),
   fPadEditor(0),
   fIgnoreSizesOnUpdate(kFALSE),
   fResetCamerasOnUpdate(kTRUE),
   fResetCamerasOnNextUpdate(kFALSE),
   fResetCameraOnDoubleClick(kTRUE)
{
   // Construct the viewer object, with following arguments:
   //    'pad' - external pad viewer is bound to
   //    'x', 'y' - initial top left position
   //    'width', 'height' - initial width/height

   InitSecondaryObjects();

   SetViewport(x, y, width, height);
}

//______________________________________________________________________________
TGLViewer::TGLViewer(TVirtualPad * pad) :
   fPad(pad),
   fContextMenu(0),
   fPerspectiveCameraXOZ(TGLVector3(1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // XOZ floor
   fPerspectiveCameraYOZ(TGLVector3(0.0, 1.0, 0.0), TGLVector3(1.0, 0.0, 0.0)), // YOZ floor
   fPerspectiveCameraXOY(TGLVector3(1.0, 0.0, 0.0), TGLVector3(0.0, 0.0,-1.0)), // XOY floor
   fOrthoXOYCamera(TGLOrthoCamera::kXOY),
   fOrthoXOZCamera(TGLOrthoCamera::kXOZ),
   fOrthoZOYCamera(TGLOrthoCamera::kZOY),
   fCurrentCamera(&fPerspectiveCameraXOZ),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fInternalRebuild(kFALSE),
   fPostSceneBuildSetup(kFALSE),
   fAcceptedAllPhysicals(kTRUE),
   fForceAcceptAll(kFALSE),
   fInternalPIDs(kFALSE),
   fNextInternalPID(1), // 0 reserved
   fComposite(0), fCSLevel(0),
   fAction(kDragNone), fLastPos(0,0), fActiveButtonID(0),
   fRedrawTimer(0),
   fClearColor(1),
   fAxesType(TGLUtil::kAxesNone),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fInitGL(kFALSE),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fAcceptedPhysicals(0),
   fRejectedPhysicals(0),
   fIsPrinting(kFALSE),
   fGLWindow(0),
   fGLDevice(fPad->GetGLDevice()),
   fGLCtxId(0),
   fPadEditor(0),
   fIgnoreSizesOnUpdate(kFALSE),
   fResetCamerasOnUpdate(kTRUE),
   fResetCamerasOnNextUpdate(kFALSE),
   fResetCameraOnDoubleClick(kTRUE)
{
   //gl-embedded viewer's ctor
   // Construct the viewer object, with following arguments:
   //    'pad' - external pad viewer is bound to
   //    'x', 'y' - initial top left position
   //    'width', 'height' - initial width/height

   InitSecondaryObjects();

   if (fGLDevice != -1) {
      // For the moment instantiate a fake context identity.
      fGLCtxId = new TGLContextIdentity;
      fGLCtxId->AddRef(0);
      Int_t viewport[4] = {0};
      gGLManager->ExtractViewport(fGLDevice, viewport);
      SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
   }
}

//______________________________________________________________________________
void TGLViewer::InitSecondaryObjects()
{
   // Common initialization.

   AddScene(&fScene);

   fLightSet = new TGLLightSet;
   fClipSet  = new TGLClipSet;  fOverlay.push_back(fClipSet);

   fSelectedPShapeRef = new TGLManipSet; fOverlay.push_back(fSelectedPShapeRef);
   fSelectedPShapeRef->SetDrawBBox(kTRUE);

   fCameraMarkup = new TGLCameraMarkupStyle;

   fRedrawTimer = new TGLRedrawTimer(*this);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
   // Destroy viewer object.

   delete fLightSet;
   delete fClipSet;
   delete fSelectedPShapeRef;
   delete fCameraMarkup;

   delete fContextMenu;
   delete fRedrawTimer;
   if (fPadEditor) fPadEditor = 0;
   fPad->ReleaseViewer3D();
   if (fGLDevice != -1)
      fGLCtxId->Release(0);
}

//______________________________________________________________________________
Bool_t TGLViewer::PreferLocalFrame() const
{
   // Indicate if viewer prefers to receive logical shape descriptions
   // in local (kTRUE) or world frame (kFALSE). For GL viewer is kTRUE always
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture
   return kTRUE;
}

void TGLViewer::PadPaint(TVirtualPad* pad)
{
   TVirtualPad *padsav = gPad;
   gPad = pad;

   TList       *prims = pad->GetListOfPrimitives();
   TObjOptLink *lnk   = (prims) ? (TObjOptLink*)prims->FirstLink() : 0;
   BeginScene();
   while (lnk)
   {
      TObject *obj = lnk->GetObject();
      if (obj->InheritsFrom(TAtt3D::Class()))
      {
         //printf("normal-painting %s / %s\n", obj->GetName(), obj->ClassName());
         obj->Paint(lnk->GetOption());
      }
      else if (obj->InheritsFrom(TH2::Class()))
      {
         //printf("histo 2d\n");
         TGLObject* log = new TH2GL();
         log->SetModel(obj, lnk->GetOption());
         log->SetBBox();
         fScene.AdoptLogical(*log);
         TGLMatrix mat;
         mat.RotateLF(3, 2, TMath::PiOver2());
         Float_t rgba[4] = { 1, 1, 1, 1};
         TGLPhysicalShape* phys = new TGLPhysicalShape
            (fNextInternalPID++, *log, mat, false, rgba);
         fScene.AdoptPhysical(*phys);
         
      }
      else if (obj->InheritsFrom(TF2::Class()))
      {
         //printf("func 2d\n");
         TGLObject* log = new TF2GL();
         log->SetModel(obj, lnk->GetOption());
         log->SetBBox();
         fScene.AdoptLogical(*log);
         TGLMatrix mat;
         mat.RotateLF(3, 2, TMath::PiOver2());
         Float_t rgba[4] = { 1, 1, 1, 1};
         TGLPhysicalShape* phys = new TGLPhysicalShape
            (fNextInternalPID++, *log, mat, false, rgba);
         fScene.AdoptPhysical(*phys);
      }
      else
      {
         // Handle 2D primitives here.
         // printf("TGLViewer::PadPaint skipping %p, %s, %s.\n",
         //        obj, obj->GetName(), obj->ClassName());
      }

      lnk = (TObjOptLink*)lnk->Next();
   }
   EndScene();

   gPad = padsav;
}

//______________________________________________________________________________
void TGLViewer::BeginScene()
{
   // Start building of viewer scene.
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture.

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::BeginScene", "entering.");
   }

   if (!fScene.TakeLock(kModifyLock)) {
      Error("TGLViewer::BeginScene", "could not take scene lock.");
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

      // Reset force acceptance of all
      fForceAcceptAll = kFALSE;

      // Reset camera interest to ensure we respond to
      // new scene range
      CurrentCamera().ResetInterest();
      fPostSceneBuildSetup = kTRUE;

      // External rebuilds could potentially invalidate all logical and
      // physical shapes - including any modified physicals
      // Physicals must be removed first
      destroyedPhysicals = fScene.DestroyPhysicals(kTRUE); // include modified
      if (fSmartRefresh) {
         fScene.BeginSmartRefresh();
      } else {
         destroyedLogicals = fScene.DestroyLogicals();
      }
   } else {
      // Internal rebuilds - destroy all non-modified physicals no longer of
      // interest to camera - retain logicals

      // MT !!!! this is spooky:
      // How can we retain physicals if physical id is checked in insert?
      // destroyedPhysicals = fScene.DestroyPhysicals(kFALSE, &CurrentCamera()); // excluded modified
      // Let's wipe them all! This will be fixed in two weeks anyhow.
      destroyedPhysicals = fScene.DestroyPhysicals(kTRUE);
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
      fScene.DumpMapSizes();
   }
}

//______________________________________________________________________________
void TGLViewer::EndScene()
{
   // End building of viewer scene
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   if (!fInternalRebuild) {
      if (fSmartRefresh) {
         fScene.EndSmartRefresh();
      }
   }
   fScene.IncTimeStamp();
   fScene.ReleaseLock(kModifyLock);

   if (fPostSceneBuildSetup) {
      PostSceneBuildSetup(fResetCamerasOnNextUpdate || fResetCamerasOnUpdate);
      fResetCamerasOnNextUpdate = kFALSE;

      // We leave fPostSceneBuildSetup set true as we want
      // another full setup after first internal rebuild
      // when we have the full scene limits
   }

   // Externally triggered scene rebuild (first pass) completed
   if (fInternalRebuild) {
      fInternalRebuild = kFALSE;
      // No more setup done after first internal scene rebuild
      fPostSceneBuildSetup = kFALSE;
   }
   RequestDraw();

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::EndScene",
           "Added %d, rejected %d physicals, accepted all:%s",
           fAcceptedPhysicals, fRejectedPhysicals, fAcceptedAllPhysicals ? "Yes":"No");
      fScene.DumpMapSizes();
   }
}

//______________________________________________________________________________
Int_t TGLViewer::AddObject(const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using internal physical IDs
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   // If this is called we are generating internal physical IDs
   fInternalPIDs = kTRUE;
   Int_t sections = AddObject(fNextInternalPID, buffer, addChildren);
   return sections;
}

//______________________________________________________________________________
// TODO: Cleanup addChildren to UInt_t flag for full termination - how returned?
Int_t TGLViewer::AddObject(UInt_t physicalID, const TBuffer3D & buffer, Bool_t * addChildren)
{
   // Add an object to the viewer, using an external physical ID
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

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

   // Assume children are always sent initially
   if (addChildren) {
      *addChildren = kTRUE;
   }

   // Scene should be modify locked
   if (fScene.CurrentLock() != kModifyLock) {
      Error("TGLViewer::AddObject", "expected scene to be modify-locked.");
      return TBuffer3D::kNone;
   }

   // Note that 'object' here is really a physical/logical pair described
   // in buffer + physical ID.

   // If adding component to a current partial composite do this now
   if (fComposite) {
      RootCsg::TBaseMesh *newMesh = RootCsg::ConvertToMesh(buffer);
      // Solaris CC can't create stl pair with enumerate type
      fCSTokens.push_back(std::make_pair(static_cast<UInt_t>(TBuffer3D::kCSNoOp), newMesh));
      return TBuffer3D::kNone;
   }

   // TODO: Could be static and save possible double lookup?
   TGLPhysicalShape * physical = fScene.FindPhysical(physicalID);
   TGLLogicalShape  * logical  = 0;

   // If we have a valid (non-zero) ID in buffer see if the logical is already cached
   if (buffer.fID) {
      logical = fScene.FindLogical(buffer.fID);
      // If not, attempt direct rendering via <ClassName>GL object.
      if (logical == 0) {
         logical = AttemptDirectRenderer(buffer.fID);
      }
   } else if (!fForceAcceptAll) {
      // If client is passing zero fID buffers we need to force accepting of all
      // so scene is never rebuilt (we can't detect cached items). Client
      // can't mix objects in scene with external or zero ids
      fForceAcceptAll = kTRUE;
      if (fNextInternalPID > 1) {
         Error("TGLViewer::AddObject", "zero fID objects can't be mixed with non-zero ones");
      }
   }

   // Function can be called twice if extra buffer filling for logical
   // is required - record last physical ID to detect
   static UInt_t lastPID = 0;

   // First attempt to add this physical
   if (physicalID != lastPID) {
      // Existing physical
      if (physical) {
         // If we have physical we should have logical cached too
         if (!logical) {
            Error("TGLViewer::AddObject", "cached physical with no assocaited cached logical");
         }

         // For external PIDs we check child interest as we may have reject children previously
         // with a different camera configuration
         if (addChildren && !fInternalPIDs) {
            *addChildren = kTRUE;
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
         if (!fForceAcceptAll) {
            // First test interest in camera - requires a bounding box
            TGLBoundingBox box;

            // If already have logical use it's BB
            if (logical) {
               box = logical->BoundingBox();
            }
            // else if bounding box in buffer valid use this
            else if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
               box.Set(buffer.fBBVertex);

            // otherwise we need to use raw points to build a bounding box with
            // If raw sections not set it will be requested by ValidateObjectBuffer
            // below and we will re-enter here
            } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
               box.SetAligned(buffer.NbPnts(), buffer.fPnts);
            }

            // Box is valid?
            if (!box.IsEmpty()) {
               // Test transformed box with camera
               box.Transform(TGLMatrix(buffer.fLocalMaster));
               Bool_t ignoreSize = fIgnoreSizesOnUpdate || !logical || logical->IgnoreSizeForOfInterest();
               Bool_t ofInterest = CurrentCamera().OfInterest(box, ignoreSize);

               // For external PID request children if physical of interest
               if (addChildren &&!fInternalPIDs) {
                  *addChildren = ofInterest;
               }

               // Physical is of interest? If not record rejection
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
      }

      // Need any extra sections in buffer?
      // If we have logical already we don't need to check raw sections
      Int_t extraSections = ValidateObjectBuffer(buffer,
                                                 logical == 0); // Check raw?
      if (extraSections != TBuffer3D::kNone) {
         return extraSections;
      } else {
         lastPID = physicalID; // Will not to re-test interest
      }
   }

   if(lastPID != physicalID)
   {
      Error("TGLViewer::AddObject", "internal physical ID tracking error?");
   }
   // By now we should need to add a physical at least
   if (physical) {
      Error("TGLViewer::AddObject", "expecting to require physical");
      return TBuffer3D::kNone;
   }

   // Create logical if required
   if (!logical) {
      logical = CreateNewLogical(buffer);
      if (!logical) {
         Error("TGLViewer::AddObject", "failed to create logical");
         return TBuffer3D::kNone;
      }
      // Add logical to scene
      fScene.AdoptLogical(*logical);
   }

   // Finally create the physical, binding it to the logical, and add to scene
   physical = CreateNewPhysical(physicalID, buffer, *logical);

   if (physical) {
      fScene.AdoptPhysical(*physical);
      buffer.fPhysicalID = physicalID; // !!! MT: should be in scene ... but so should all add object stuff
      ++fAcceptedPhysicals;
      if (gDebug>3 && fAcceptedPhysicals%1000 == 0) {
         Info("TGLViewer::AddObject", "added %d physicals", fAcceptedPhysicals);
      }
   } else {
      Error("TGLViewer::AddObject", "failed to create physical");
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
   // Open new composite container.
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture.

   if (fComposite) {
      Error("TGLViewer::OpenComposite", "composite already open");
      return kFALSE;
   }
   UInt_t extraSections = AddObject(buffer, addChildren);
   if (extraSections != TBuffer3D::kNone) {
      Error("TGLViewer::OpenComposite", "expected top level composite to not require extra buffer sections");
   }

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
   // Close composite container
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   // If we have a partially complete composite build it now
   if (fComposite) {
      // TODO: Why is this member and here - only used in BuildComposite()
      fCSLevel = 0;

      RootCsg::TBaseMesh *resultMesh = BuildComposite();
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
   // Add composite operation used to combine objects added via AddObject
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   fCSTokens.push_back(std::make_pair(operation, (RootCsg::TBaseMesh *)0));
}


//______________________________________________________________________________
Bool_t TGLViewer::RebuildScene()
{
   // If we accepted all offered physicals into the scene no point in
   // rebuilding it.

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

   // Internally triggered scene rebuild
   fInternalRebuild = kTRUE;

   TGLStopwatch timer;
   if (gDebug>2 || fDebugMode) {
      timer.Start();
   }

   // Request a scene fill
   // TODO: Just marking modified doesn't seem to result in pad repaint - need to check on
   // MT: This will be obsolete soon.
   PadPaint(fPad);

   if (gDebug>2 || fDebugMode) {
      Info("TGLViewer::RebuildScene", "rebuild complete in %f", timer.End());
   }

   return kTRUE;
}

//______________________________________________________________________________
Int_t TGLViewer::ValidateObjectBuffer(const TBuffer3D & buffer, Bool_t includeRaw) const
{
   // Validate if the passed 'buffer' contains all sections we require to add object.
   // Returns Int_t combination of TBuffer::ESection flags still required - or
   // TBuffer3D::kNone if buffer is valid.
   // If 'includeRaw' is kTRUE check for kRaw/kRawSizes - skip otherwise.
   // See base/src/TVirtualViewer3D.cxx for description of viewer architecture

   // kCore: Should always be filled
   if (!buffer.SectionsValid(TBuffer3D::kCore)) {
      Error("TGLViewer::ValidateObjectBuffer", "kCore section of buffer should be filled always");
      return TBuffer3D::kNone;
   }

   // Need to check raw (kRaw/kRawSizes)?
   if (!includeRaw) {
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
         Error("TGLViewer::ValidateObjectBuffer", "failed to cast buffer of type 'kSphere' to TBuffer3DSphere");
         return TBuffer3D::kNone;
      }
   }
   // 3. kBoundingBox is not filled - we generate a bounding box from
   else if (!buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      needRaw = kTRUE;
   }
   // 4. kShapeSpecific is not filled - except in case of top level composite
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
   // Create and return a new TGLLogicalShape from the supplied buffer
   TGLLogicalShape * newLogical = 0;

   if (buffer.fColor == 1) // black -> light-brown; std behaviour for geom
      const_cast<TBuffer3D&>(buffer).fColor = 42;

   switch (buffer.Type()) {
   case TBuffer3DTypes::kLine:
      newLogical = new TGLPolyLine(buffer);
      break;
   case TBuffer3DTypes::kMarker:
      newLogical = new TGLPolyMarker(buffer);
      break;
   case TBuffer3DTypes::kSphere: {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         // We can only draw solid uncut spheres natively at present
         if (sphereBuffer->IsSolidUncut()) {
            newLogical = new TGLSphere(*sphereBuffer);
         } else {
            newLogical = new TGLFaceSet(buffer);
         }
      }
      else {
         Error("TGLViewer::CreateNewLogical", "failed to cast buffer of type 'kSphere' to TBuffer3DSphere");
      }
      break;
   }
   case TBuffer3DTypes::kTube:
   case TBuffer3DTypes::kTubeSeg:
   case TBuffer3DTypes::kCutTube: {
      const TBuffer3DTube * tubeBuffer = dynamic_cast<const TBuffer3DTube *>(&buffer);
      if (tubeBuffer)
      {
         newLogical = new TGLCylinder(*tubeBuffer);
      }
      else {
         Error("TGLViewer::CreateNewLogical", "failed to cast buffer of type 'kTube/kTubeSeg/kCutTube' to TBuffer3DTube");
      }
      break;
   }
   case TBuffer3DTypes::kComposite: {
      // Create empty faceset and record partial complete composite object
      // Will be populated with mesh in CloseComposite()
      if (fComposite) {
         Error("TGLViewer::CreateNewLogical", "composite already open");
      }
      fComposite = new TGLFaceSet(buffer);
      newLogical = fComposite;
      break;
   }
   default:
      newLogical = new TGLFaceSet(buffer);
      break;
   }

   return newLogical;
}

//______________________________________________________________________________
TGLPhysicalShape*
TGLViewer::CreateNewPhysical(      UInt_t            ID,
                             const TBuffer3D       & buffer,
                             const TGLLogicalShape & logical) const
{
   // Create and return a new TGLPhysicalShape with id 'ID', using
   // 'buffer' placement information (translation etc), and bound to
   // suppled 'logical'

   // Extract indexed color from buffer
   // TODO: Still required? Better use proper color triplet in buffer?
   Int_t colorIndex = buffer.fColor;
   if (colorIndex < 0) colorIndex = 42;
   Float_t rgba[4];
   TGLScene::RGBAFromColorIdx(rgba, colorIndex, buffer.fTransparency);
   return new TGLPhysicalShape(ID, logical, buffer.fLocalMaster,
                               buffer.fReflection, rgba);
}

//______________________________________________________________________________
RootCsg::TBaseMesh *TGLViewer::BuildComposite()
{
   // Build and return composite shape mesh
   const CSPart_t &currToken = fCSTokens[fCSLevel];
   UInt_t opCode = currToken.first;

   if (opCode != TBuffer3D::kCSNoOp) {
      ++fCSLevel;
      RootCsg::TBaseMesh *left = BuildComposite();
      RootCsg::TBaseMesh *right = BuildComposite();
      //RootCsg::TBaseMesh *result = 0;
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


/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::UpdateScene()
{
   // Force a scene update.
   // Code segments taken from protected RebuildScene().

   // We are going to rebuild the scene - ensure any pending redraw timer cancelled now
   fRedrawTimer->Stop();

   // Pretend the update request came from outside.
   fInternalRebuild = kFALSE;

   PadPaint(fPad);
}

//______________________________________________________________________________
void TGLViewer::ResetCurrentCamera()
{
   // Resets position/rotation of current camera to default values.

   CurrentCamera().Reset();
}

//______________________________________________________________________________
void TGLViewer::SetupCameras(Bool_t reset)
{
   // Setup cameras for current bounding box.

   if (IsLocked()) {
      Error("TGLViewer::SetupCameras", "expected kUnlocked, found %s", LockName(CurrentLock()));
      return;
   }

   // Setup cameras if scene box is not empty
   const TGLBoundingBox & box = fOverallBoundingBox;
   if (!box.IsEmpty()) {
      fPerspectiveCameraYOZ.Setup(box, reset);
      fPerspectiveCameraXOZ.Setup(box, reset);
      fPerspectiveCameraXOY.Setup(box, reset);
      fOrthoXOYCamera.Setup(box, reset);
      fOrthoXOZCamera.Setup(box, reset);
      fOrthoZOYCamera.Setup(box, reset);
   }
}

//______________________________________________________________________________
void TGLViewer::PostSceneBuildSetup(Bool_t resetCameras)
{
   // Perform post scene (re)build setup

   fOverallBoundingBox = fScene.BoundingBox();
   SetupCameras(resetCameras);

   // Set default reference to center
   fReferencePos.Set(fOverallBoundingBox.Center());
}


/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::InitGL()
{
   // Initialise GL state if not already done
   if (fInitGL) {
      Error("TGLViewer::InitGL", "GL already initialised");
   }

   // GL initialisation
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glClearColor(0.f, 0.f, 0.f, 1.f);
   glClearDepth(1.0);
   glMaterialf(GL_BACK, GL_SHININESS, 0.0);
   glPolygonMode(GL_FRONT, GL_FILL);
   glDisable(GL_BLEND);

   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodelAmb);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

   TGLUtil::CheckError("TGLViewer::InitGL");
   fInitGL = kTRUE;
}

//______________________________________________________________________________
void TGLViewer::RequestDraw(Short_t LOD)
{
   // Post request for redraw of viewer at level of detail 'LOD'
   // Request is directed via cross thread gVirtualGL object.

   fRedrawTimer->Stop();
   // Ignore request if GL window or context not yet availible - we
   // will get redraw later
   if (!fGLWindow && fGLDevice == -1) {
      fRedrawTimer->RequestDraw(100, LOD);
      return;
   }

   // Take scene draw lock - to be revisited
   if ( ! TakeLock(kDrawLock)) {
      // If taking drawlock fails the previous draw is still in progress
      // set timer to do this one later
      if (gDebug>3) {
         Info("TGLViewer::RequestDraw", "viewer locked - requesting another draw.");
      }
      fRedrawTimer->RequestDraw(100, LOD);
      return;
   }
   fLOD = LOD;

   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLViewer *)0x%x)->DoDraw()", this));
   else
      DoDraw();
}

//______________________________________________________________________________
void TGLViewer::PreRender()
{
   fCamera = fCurrentCamera;
   fClip   = fClipSet->GetCurrentClip();
   if (fGLDevice != -1)
   {
      fRnrCtx->SetGLCtxIdentity(fGLCtxId);
      fGLCtxId->DeleteDisplayLists();
   }
   TGLViewerBase::PreRender();
   // Setup lighting
   fLightSet->StdSetupLights(fOverallBoundingBox, *fCamera, fDebugMode);
   fClipSet->SetupClips(fOverallBoundingBox);
}

//______________________________________________________________________________
void TGLViewer::DoDraw()
{
   // Draw out the the current viewer/scene

   // Locking mainly for Win32 multi thread safety - but no harm in all using it
   // During normal draws a draw lock is taken in other thread (Win32) in RequestDraw()
   // to ensure thread safety. For PrintObjects repeated Draw() calls are made.
   // If no draw lock taken get one now.

   fRedrawTimer->Stop();

   if (CurrentLock() != kDrawLock) {
      if ( ! TakeLock(kDrawLock)) {
         Error("TGLViewer::DoDraw", "viewer is %s", LockName(CurrentLock()));
         return;
      }
   }

   if (fGLDevice != -1) {
      Int_t viewport[4] = {};
      gGLManager->ExtractViewport(fGLDevice, viewport);
      SetViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
   }

   TGLStopwatch timer;
   if (gDebug>2) {
      timer.Start();
   }

   // GL pre draw setup
   if (!fIsPrinting) PreDraw();

   PreRender();

   // Setup total scene draw time
   // Unlimted for high quality draws, 200 msec otherwise
   Double_t sceneDrawTime = (fLOD == TGLRnrCtx::kLODHigh) ? 0.0 : 200.0;
   sceneDrawTime /= fVisScenes.size();
   fRnrCtx->SetRenderTimeout(sceneDrawTime);

   Render();

   DrawGuides();
   glClear(GL_DEPTH_BUFFER_BIT);
   RenderOverlay();
   DrawCameraMarkup();
   DrawDebugInfo();

   PostRender();
   PostDraw();

   ReleaseLock(kDrawLock);

   if (gDebug>2) {
      Info("TGLViewer::DoDraw()", "Took %f msec", timer.End());
   }

   Bool_t redrawReq = kFALSE;

   // Debug mode have forced rebuilds only
   if (!fDebugMode) {
      // Final draw pass
      if (fLOD == TGLRnrCtx::kLODHigh) {
         RebuildScene();
      } else {
         // Final draw pass required
         redrawReq = kTRUE;
      }
   } else {
      // Final draw pass required?
      redrawReq = fLOD != TGLRnrCtx::kLODHigh;
   }

   // Request final pass high quality redraw via timer
   if (redrawReq) {
      fRedrawTimer->RequestDraw(100, TGLRnrCtx::kLODHigh);
   }

}

//______________________________________________________________________________
void TGLViewer::DrawGuides()
{
   // Draw reference marker and coordinate axes.

   glDisable(GL_DEPTH_TEST);

   if (fReferenceOn)
      TGLUtil::DrawReferenceMarker(*fCamera, fReferencePos);

   if (fAxesType != TGLUtil::kAxesOrigin)
      glEnable(GL_DEPTH_TEST);

   TGLUtil::DrawSimpleAxes(*fCamera, fOverallBoundingBox, fAxesType);

   glEnable(GL_DEPTH_TEST);
}

//______________________________________________________________________________
void TGLViewer::DrawCameraMarkup()
{
   // Draw camera markup overlay.

   if (fCameraMarkup && fCameraMarkup->Show())
   {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      const TGLRect& vp = fRnrCtx->RefCamera().RefViewport();
      gluOrtho2D(0., vp.Width(), 0., vp.Height());
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();
      glDisable(GL_LIGHTING);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glDisable(GL_DEPTH_TEST);
      fRnrCtx->RefCamera().Markup(fCameraMarkup);
      glEnable(GL_DEPTH_TEST);
      glEnable(GL_LIGHTING);
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
   }
}

//______________________________________________________________________________
void TGLViewer::DrawDebugInfo()
{
   // If in debug mode draw camera aids and overall bounding box.

   if (fDebugMode)
   {
      glDisable(GL_LIGHTING);
      CurrentCamera().DrawDebugAids();

      // Green scene bounding box
      glColor3d(0.0, 1.0, 0.0);
      fOverallBoundingBox.Draw();

      // Scene bounding box center sphere (green) and
      glDisable(GL_DEPTH_TEST);
      Double_t size = fOverallBoundingBox.Extents().Mag() / 200.0;
      static Float_t white[4] = {1.0, 1.0, 1.0, 1.0};
      TGLUtil::DrawSphere(TGLVertex3(0.0, 0.0, 0.0), size, white);
      static Float_t green[4] = {0.0, 1.0, 0.0, 1.0};
      const TGLVertex3 & center = fOverallBoundingBox.Center();
      TGLUtil::DrawSphere(center, size, green);
      glEnable(GL_DEPTH_TEST);

      glEnable(GL_LIGHTING);
   }
}

//______________________________________________________________________________
void TGLViewer::PreDraw()
{
   // Perform GL work which must be done before each draw of scene
   MakeCurrent();
   // Initialise GL if not done
   if (!fInitGL) {
      InitGL();
   }

   // For embedded gl clear color must be pad's background color.
   Color_t ci = (fGLDevice != -1) ? gPad->GetFillColor() : fClearColor;
   TColor *color = gROOT->GetColor(ci);
   Float_t sc[3] = {1.f, 1.f, 1.f};
   if (color)
      color->GetRGB(sc[0], sc[1], sc[2]);
   glClearColor(sc[0], sc[1], sc[2], 1.);

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   TGLUtil::CheckError("TGLViewer::PreDraw");
}

//______________________________________________________________________________
void TGLViewer::PostDraw()
{
   // Perform GL work which must be done after each draw of scene
   glFlush();
   SwapBuffers();

   // Flush everything in case picking starts
   //   glFlush();

   TGLUtil::CheckError("TGLViewer::PostDraw");
}

//______________________________________________________________________________
void TGLViewer::MakeCurrent() const
{
   // Make GL context current
   if (fGLDevice == -1)
      fGLWindow->MakeCurrent();
   else gGLManager->MakeCurrent(fGLDevice);

   // Don't call TGLUtil::CheckError() as we do not
   // have to be in GL thread here - GL window will call
   // via gVirtualGL. Again re-enable once TGLManager replaces
   // TGLUtil::CheckError();
}

//______________________________________________________________________________
void TGLViewer::SwapBuffers() const
{
   // Swap GL buffers
   if ( ! IsDrawOrSelectLock()) {
      Error("TGLViewer::SwapBuffers", "viewer is %s", LockName(CurrentLock()));
   }
   if (fGLDevice == -1)
      fGLWindow->SwapBuffers();
   else {
      gGLManager->ReadGLBuffer(fGLDevice);
      gGLManager->Flush(fGLDevice);
      gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
   }
}

//______________________________________________________________________________
Bool_t TGLViewer::RequestSelect(Int_t x, Int_t y, Bool_t trySecSel)
{
   // Post request for select draw of viewer, picking objects round the WINDOW
   // point (x,y).
   // Request is directed via cross thread gVirtualGL object

   // Take select lock on scene immediately we enter here - it is released
   // in the other (drawing) thread - see TGLViewer::Select()
   // Removed when gVirtualGL removed

   if ( ! TakeLock(kSelectLock)) {
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLViewer *)0x%x)->DoSelect(%d, %d, %s)", this, x, y, trySecSel ? "kTRUE" : "kFALSE")));
   else
      return DoSelect(x, y, trySecSel);
}

//______________________________________________________________________________
Bool_t TGLViewer::DoSelect(Int_t x, Int_t y, Bool_t trySecSel)
{
   // Perform GL selection, picking objects overlapping WINDOW
   // area described by 'rect'. Return kTRUE if selection should be
   // changed, kFALSE otherwise.
   // Select lock should already been taken in other thread in
   // TGLViewer::ReqSelect().

   if (CurrentLock() != kSelectLock) {
      Error("TGLViewer::DoSelect", "expected kSelectLock, found %s", LockName(CurrentLock()));
      return kFALSE;
   }

   MakeCurrent();

   fRnrCtx->BeginSelection(x, y, 3);
   glRenderMode(GL_SELECT);

   PreRender();
   Render();
   PostRender();

   Int_t nHits = glRenderMode(GL_RENDER);
   fRnrCtx->EndSelection(nHits);

   // Process selection.
   if (gDebug > 0) Info("TGLViewer::DoSelect", "Primary select nHits=%d.", nHits);

   if (nHits > 0)
   {
      Int_t idx = 0;
      if (FindClosestRecord(fSelRec, idx))
      {
         if (fSelRec.GetTransparent())
         {
            TGLSelectRecord opaque;
            if (FindClosestOpaqueRecord(opaque, ++idx))
               fSelRec = opaque;
         }
         if (gDebug > 1) fSelRec.Print();
      }
   } else {
      fSelRec.Reset();
   }

   if ( ! trySecSel)
   {
      ReleaseLock(kSelectLock);
      return ! TGLSelectRecord::AreSameSelectionWise(fSelRec, fCurrentSelRec);
   }

   //  Secondary selection.
   {
      if ( nHits < 1 || ! fSelRec.GetSceneInfo() || ! fSelRec.GetPhysShape() ||
           ! fSelRec.GetPhysShape()->GetLogical()->SupportsSecondarySelect())
      {
         if (gDebug > 0)
            Info("TGLViewer::DoSelect", "Skipping secondary selection "
                 "(nPrimHits=%d, sinfo=0x%lx, pshape=0x%lx).\n",
                 nHits, fSelRec.GetSceneInfo(), fSelRec.GetPhysShape());
         ReleaseLock(kSelectLock);
         fSecSelRec.Reset();
         return kFALSE;
      }

      TGLSceneInfo*    sinfo = fSelRec.GetSceneInfo();
      TGLSceneBase*    scene = sinfo->GetScene();
      TGLPhysicalShape* pshp = fSelRec.GetPhysShape();

      SceneInfoList_t foo;
      foo.push_back(sinfo);
      fScenes.swap(foo);
      fRnrCtx->BeginSelection(x, y, 3);
      fRnrCtx->SetSecSelection(kTRUE);
      glRenderMode(GL_SELECT);

      PreRender();
      fRnrCtx->SetSceneInfo(sinfo);
      scene->PreRender(*fRnrCtx);
      fRnrCtx->SetDrawPass(TGLRnrCtx::kPassFill);
      fRnrCtx->SetShapeLOD(TGLRnrCtx::kLODHigh);
      glPushName(pshp->ID());
      // !!! Hack: does not use clipping and proper draw-pass settings.
      pshp->Draw(*fRnrCtx);
      glPopName();
      scene->PostRender(*fRnrCtx);
      fRnrCtx->SetSceneInfo(0);
      PostRender();

      Int_t nSecHits = glRenderMode(GL_RENDER);
      fRnrCtx->EndSelection(nSecHits);
      fScenes.swap(foo);

      if (gDebug > 0) Info("TGLViewer::DoSelect", "Secondary select nSecHits=%d.", nSecHits);

      ReleaseLock(kSelectLock);

      if (nSecHits > 0)
      {
         fSecSelRec = fSelRec;
         fSecSelRec.SetRawOnly(fRnrCtx->GetSelectBuffer()->RawRecord(0));
         if (gDebug > 1) fSecSelRec.Print();
         return kTRUE;
      } else {
         fSecSelRec.Reset();
         return kFALSE;
      }
   }
}

//______________________________________________________________________________
void TGLViewer::ApplySelection()
{
   // Process result from last selection (in fSelRec) and
   // extract a new current selection from it.
   // Here we only use physical shape.

   fCurrentSelRec = fSelRec;

   TGLPhysicalShape * selPhys = fSelRec.GetPhysShape();
   fSelectedPShapeRef->SetPShape(selPhys);

   // Inform external client selection has been modified.
   SelectionChanged();

   RequestDraw(TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
Bool_t TGLViewer::RequestOverlaySelect(Int_t x, Int_t y)
{
   // Post request for select draw of viewer, picking objects round the WINDOW
   // point (x,y).
   // Request is directed via cross thread gVirtualGL object

   // Take select lock on scene immediately we enter here - it is released
   // in the other (drawing) thread - see TGLViewer::Select()
   // Removed when gVirtualGL removed

   if ( ! TakeLock(kSelectLock)) {
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLViewer *)0x%x)->DoSelect(%d, %d)", this, x, y)));
   else
      return DoOverlaySelect(x, y);
}

//______________________________________________________________________________
Bool_t TGLViewer::DoOverlaySelect(Int_t x, Int_t y)
{
   // Perform GL selection, picking overlay objects only.
   // Return TRUE if the selected overlay-element has changed.

   if (CurrentLock() != kSelectLock) {
      Error("TGLViewer::DoOverlaySelect", "expected kSelectLock, found %s", LockName(CurrentLock()));
      return kFALSE;
   }

   MakeCurrent();

   fRnrCtx->BeginSelection(x, y, 3);
   glRenderMode(GL_SELECT);

   PreRenderOverlaySelection();
   RenderOverlay();
   PostRenderOverlaySelection();

   Int_t nHits = glRenderMode(GL_RENDER);
   fRnrCtx->EndSelection(nHits);

   // Process overlay selection.
   TGLOverlayElement * selElm = 0;
   if (nHits > 0)
   {
      Int_t idx = 0;
      while (idx < nHits && FindClosestOverlayRecord(fOvlSelRec, idx))
      {
         TGLOverlayElement* el = fOvlSelRec.GetOvlElement();
         if (el == fCurrentOvlElm)
         {
            if (el->MouseStillInside(fOvlSelRec))
            {
               selElm = el;
               break;
            }
         }
         else if (el->MouseEnter(fOvlSelRec))
         {
            selElm = el;
            break;
         }
      }
   }
   else
   {
      fOvlSelRec.Reset();
   }

   ReleaseLock(kSelectLock);

   if (fCurrentOvlElm != selElm)
   {
      if (fCurrentOvlElm) fCurrentOvlElm->MouseLeave();
      fCurrentOvlElm = selElm;
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

/**************************************************************************/
// Viewport
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::SetViewport(Int_t x, Int_t y, Int_t width, Int_t height)
{
   // Set viewer viewport (window area) with bottom/left at (x,y), with
   // dimensions 'width'/'height'

   if (IsLocked() && fGLDevice == -1) {
      Error("TGLViewer::SetViewport", "expected kUnlocked, found %s", LockName(CurrentLock()));
      return;
   }
   // Only process if changed
   if (fViewport.X() == x && fViewport.Y() == y &&
       fViewport.Width() == width && fViewport.Height() == height) {
      return;
   }

   fViewport.Set(x, y, width, height);
   fCurrentCamera->SetViewport(fViewport);

   // Request redraw via timer as window resize can result in stream of calls
   // RequestDraw(TGLRnrCtx::kLODMed);
   if (gDebug>2) {
      Info("TGLViewer::SetViewport", "updated - corner %d,%d dimensions %d,%d", x, y, width, height);
   }
}


/**************************************************************************/
// Camera methods
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::SetCurrentCamera(ECameraType cameraType)
{
   // Set current active camera - 'cameraType' one of:
   // kCameraPerspX, kCameraPerspY, kCameraPerspZ
   // kCameraOrthoXOY, kCameraOrthoXOZ, kCameraOrthoZOY

   if (IsLocked()) {
      Error("TGLViewer::SetCurrentCamera", "expected kUnlocked, found %s", LockName(CurrentLock()));
      return;
   }

   // TODO: Move these into a vector!
   switch(cameraType) {
      case(kCameraPerspXOZ): {
         fCurrentCamera = &fPerspectiveCameraXOZ;
         break;
      }
      case(kCameraPerspYOZ): {
         fCurrentCamera = &fPerspectiveCameraYOZ;
         break;
      }
      case(kCameraPerspXOY): {
         fCurrentCamera = &fPerspectiveCameraXOY;
         break;
      }
      case(kCameraOrthoXOY): {
         fCurrentCamera = &fOrthoXOYCamera;
         break;
      }
      case(kCameraOrthoXOZ): {
         fCurrentCamera = &fOrthoXOZCamera;
         break;
      }
      case(kCameraOrthoZOY): {
         fCurrentCamera = &fOrthoZOYCamera;
         break;
      }
      default: {
         Error("TGLViewer::SetCurrentCamera", "invalid camera type");
         break;
      }
   }

   // Ensure any viewport has been propigated to the current camera
   fCurrentCamera->SetViewport(fViewport);
   RefreshPadEditor(this);

   // And viewer is redrawn
   RequestDraw(TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
void TGLViewer::SetOrthoCamera(ECameraType camera,
                               Double_t left, Double_t right,
                               Double_t top,  Double_t bottom)
{
   // Set an orthographic camera to supplied configuration - note this
   // does not need to be the current camera - though you will not see
   // the effect if it is not.
   //
   // 'camera' defines the ortho camera - one of kCameraOrthoXOY / XOZ / ZOY
   // 'left' / 'right' / 'top' / 'bottom' define the WORLD coordinates which
   // corresepond with the left/right/top/bottom positions on the GL viewer viewport
   // E.g. for kCameraOrthoXOY camera left/right are X world coords,
   // top/bottom are Y world coords
   // As this is an orthographic camera the other axis (in eye direction) is
   // no relevant. The near/far clip planes are set automatically based in scene
   // contents

   // TODO: Move these into a vector!
   switch(camera) {
      case(kCameraOrthoXOY): {
         fOrthoXOYCamera.Configure(left, right, top, bottom);
         if (fCurrentCamera == &fOrthoXOYCamera) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case(kCameraOrthoXOZ): {
         fOrthoXOZCamera.Configure(left, right, top, bottom);
         if (fCurrentCamera == &fOrthoXOZCamera) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case(kCameraOrthoZOY): {
         fOrthoZOYCamera.Configure(left, right, top, bottom);
         if (fCurrentCamera == &fOrthoZOYCamera) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      default: {
         Error("TGLViewer::SetOrthoCamera", "invalid camera type");
         break;
      }
   }
}

//______________________________________________________________________________
void TGLViewer::SetPerspectiveCamera(ECameraType camera,
                                     Double_t fov, Double_t dolly,
                                     Double_t center[3],
                                     Double_t hRotate, Double_t vRotate)
{
   // Set a perspective camera to supplied configuration - note this
   // does not need to be the current camera - though you will not see
   // the effect if it is not.
   //
   // 'camera' defines the persp camera - one of kCameraPerspXOZ, kCameraPerspYOZ, kCameraPerspXOY
   // 'fov' - field of view (lens angle) in degrees (clamped to 0.1 - 170.0)
   // 'dolly' - distance from 'center'
   // 'center' - world position from which dolly/hRotate/vRotate are measured
   //             camera rotates round this, always facing in (in center of viewport)
   // 'hRotate' - horizontal rotation from initial configuration in degrees
   // 'hRotate' - vertical rotation from initial configuration in degrees

   // TODO: Move these into a vector!
   switch(camera) {
      case(kCameraPerspXOZ): {
         fPerspectiveCameraXOZ.Configure(fov, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fPerspectiveCameraXOZ) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case(kCameraPerspYOZ): {
         fPerspectiveCameraYOZ.Configure(fov, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fPerspectiveCameraYOZ) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case(kCameraPerspXOY): {
         fPerspectiveCameraXOY.Configure(fov, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fPerspectiveCameraXOY) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      default: {
         Error("TGLViewer::SetPerspectiveCamera", "invalid camera type");
         break;
      }
   }
}


/**************************************************************************/
// Guide methods
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::GetGuideState(Int_t & axesType, Bool_t & referenceOn, Double_t referencePos[3]) const
{
   // Fetch the state of guides (axes & reference markers) into arguments
   axesType    = fAxesType;
   referenceOn = fReferenceOn;
   referencePos[0] = fReferencePos.X();
   referencePos[1] = fReferencePos.Y();
   referencePos[2] = fReferencePos.Z();
}

//______________________________________________________________________________
void TGLViewer::SetGuideState(Int_t axesType, Bool_t referenceOn, const Double_t referencePos[3])
{
   // Set the state of guides (axes & reference markers) from arguments
   fAxesType    = axesType;
   fReferenceOn = referenceOn;
   fReferencePos.Set(referencePos[0], referencePos[1], referencePos[2]);
   if (fGLDevice != -1)
      gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
   RequestDraw();
}


// Selected physical
//______________________________________________________________________________
const TGLPhysicalShape * TGLViewer::GetSelected() const
{
   // Return selected physical shape.

   return fSelectedPShapeRef->GetPShape();
}

/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::SelectionChanged()
{
   // Emit signal indicating selection has changed
   Emit("SelectionChanged()");
}


/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
Int_t TGLViewer::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
   // Calcaulate and return pixel distance to nearest viewer object from
   // window location px, py
   // This is provided for use when embedding GL viewer into pad

   // Can't track the indvidual objects in rollover. Just set the viewer as the
   // selected object, and return 0 (object identified) so we receive ExecuteEvent calls
   gPad->SetSelected(this);
   return 0;
}

//______________________________________________________________________________
void TGLViewer::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Process event of type 'event' - one of EEventType types,
   // occuring at window location px, py
   // This is provided for use when embedding GL viewer into pad

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
   eventSt.fState = 0;

   if (event != kKeyPress) {
      eventSt.fY -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
      eventSt.fX -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
   }

   switch (event) {
      case kMouseMotion:
         eventSt.fCode = kMouseMotion;
         eventSt.fType = kMotionNotify;
         HandleMotion(&eventSt);
         break;
      case kButton1Down:
      case kButton1Up:
      {
         eventSt.fCode = kButton1;
         eventSt.fType = event == kButton1Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton2Down:
      case kButton2Up:
      {
         eventSt.fCode = kButton2;
         eventSt.fType = event == kButton2Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton3Down:
      {
         eventSt.fState = kKeyShiftMask;
         eventSt.fCode = kButton1;
         eventSt.fType = kButtonPress;
         HandleButton(&eventSt);
      }
      break;
      case kButton3Up:
      {
         eventSt.fCode = kButton3;
         eventSt.fType = kButtonRelease;//event == kButton3Down ? kButtonPress:kButtonRelease;
         HandleButton(&eventSt);
      }
      break;
      case kButton1Double:
      case kButton2Double:
      case kButton3Double:
      {
         eventSt.fCode = kButton1Double ? kButton1 : kButton2Double ? kButton2 : kButton3;
         eventSt.fType = kButtonDoubleClick;
         HandleDoubleClick(&eventSt);
      }
      break;
      case kButton1Motion:
      case kButton2Motion:
      case kButton3Motion:
      {

         eventSt.fCode = event == kButton1Motion ? kButton1 : event == kButton2Motion ? kButton2 : kButton3;
         eventSt.fType = kMotionNotify;
         HandleMotion(&eventSt);
      }
      break;
      case kKeyPress: // We only care about full key 'presses' not individual down/up
      {
         eventSt.fType = kGKeyPress;
         eventSt.fCode = py; // px contains key code - need modifiers from somewhere
         HandleKey(&eventSt);
      }
      break;
      case 6://trick :)
         if (CurrentCamera().Zoom(+50, kFALSE, kFALSE)) { //TODO : val static const somewhere
            if (fGLDevice != -1) {
               gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
               gVirtualX->SetDrawMode(TVirtualX::kCopy);
            }
            RequestDraw();
         }
         break;
      case 5://trick :)
         if (CurrentCamera().Zoom(-50, kFALSE, kFALSE)) { //TODO : val static const somewhere
            if (fGLDevice != -1) {
               gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
               gVirtualX->SetDrawMode(TVirtualX::kCopy);
            }
            RequestDraw();
         }
         break;
      case 7://trick :)
         eventSt.fState = kKeyShiftMask;
         eventSt.fCode = kButton1;
         eventSt.fType = kButtonPress;
         HandleButton(&eventSt);
         break;
      default:
      {
        // Error("TGLViewer::ExecuteEvent", "invalid event type");
      }
   }
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleEvent(Event_t *event)
{
   // Handle generic Event_t type 'event' - provided to catch focus changes
   // and terminate any interaction in viewer
   if (event->fType == kFocusIn) {
      if (fAction != kNone) {
         Error("TGLViewer::HandleEvent", "active action at focus in");
      }
      fAction = kDragNone;
   }
   if (event->fType == kFocusOut) {
      fAction = kDragNone;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleButton(Event_t * event)
{
   // Handle mouse button 'event'
   if (IsLocked()) {
      if (gDebug>2) {
         Info("TGLViewer::HandleButton", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   // Button DOWN
   if (event->fType == kButtonPress)
   {
      // Allow a single action/button down/up pairing - block others
      if (fAction != kNone)
         return kFALSE;

      Bool_t grabPointer = kFALSE;
      Bool_t handled     = kFALSE;

      // Record active button for release
      fActiveButtonID = event->fCode;

      if (fAction == kDragNone && fCurrentOvlElm)
      {
         if (fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event))
         {
            handled     = kTRUE;
            grabPointer = kTRUE;
            fAction     = kDragOverlay;
            RequestDraw();
         }
      }
      if ( ! handled)
      {
         switch(event->fCode)
         {
            // LEFT mouse button
            case kButton1:
            {
               if (event->fState & kKeyShiftMask) {
                  if (RequestSelect(event->fX, event->fY)) {
                     ApplySelection();
                  }
               } else if (event->fState & kKeyControlMask) {
                  RequestSelect(event->fX, event->fY, kTRUE);
                  if (fSecSelRec.GetPhysShape() != 0) {
                     TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
                        (*fSecSelRec.GetPhysShape()->GetLogical());
                     lshape.ProcessSelection(*fRnrCtx, fSecSelRec);
                  }
               } else {
                  fAction = kDragCameraRotate;
                  grabPointer = kTRUE;
               }
               break;
            }
               // MID mouse button
            case kButton2:
            {
               fAction = kDragCameraTruck;
               grabPointer = kTRUE;
               break;
            }
               // RIGHT mouse button
            case kButton3:
            {
               // Shift + Right mouse - select+context menu
               if (event->fState & kKeyShiftMask) {
                  RequestSelect(event->fX, event->fY);
                  const TGLPhysicalShape * selected = fSelRec.GetPhysShape();
                  if (selected) {
                     if (!fContextMenu) {
                        fContextMenu = new TContextMenu("glcm", "GL Viewer Context Menu");
                     }
                     Int_t    x, y;
                     Window_t childdum;
                     gVirtualX->TranslateCoordinates(fGLWindow->GetId(),
                                                     gClient->GetDefaultRoot()->GetId(),
                                                     event->fX, event->fY, x, y, childdum);
                     selected->InvokeContextMenu(*fContextMenu, x, y);
                     // MT-TODO: Find a way to request redraw after dialog has finished.
                  }
               } else {
                  fAction = kDragCameraDolly;
                  grabPointer = kTRUE;
               }
               break;
            }
         }
      }
   }
   // Button UP
   else if (event->fType == kButtonRelease)
   {
      if (fAction == kDragOverlay) {
         fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event);
         SelectionChanged();
         // XXX CLIPA ClipChanged();
         if (RequestOverlaySelect(event->fX, event->fY))
            RequestDraw();
      }

      // TODO: Check on Linux - on Win32 only see button release events
      // for mouse wheel
      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         // Note: Modifiers (ctrl/shift) disabled as fState doesn't seem to
         // have correct modifier flags with mouse wheel under Windows.
         case(kButton5): {
            // Zoom out (adjust camera FOV)
            if (CurrentCamera().Zoom(+50, kFALSE, kFALSE)) { //TODO : val static const somewhere
               RequestDraw();
            }
            break;
         }
         case(kButton4): {
            // Zoom in (adjust camera FOV)
            if (CurrentCamera().Zoom(-50, kFALSE, kFALSE)) { //TODO : val static const somewhere
               RequestDraw();
            }
            break;
         }
      }
      fAction = kDragNone;
      if (fGLDevice != -1)
         gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleDoubleClick(Event_t *event)
{
   // Handle mouse double click 'event'
   if (IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleDoubleClick", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   // Reset interactive camera mode on button double
   // click (unless mouse wheel)
   if (event->fCode != kButton4 && event->fCode != kButton5) {
      if (fResetCameraOnDoubleClick) {
         ResetCurrentCamera();
         RequestDraw();
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleConfigureNotify(Event_t *event)
{
   // Handle configure notify 'event' - a window resize/movement
   if (IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleConfigureNotify", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   if (event) {
      SetViewport(event->fX, event->fY, event->fWidth, event->fHeight);
      RequestDraw(TGLRnrCtx::kLODMed);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleKey(Event_t *event)
{
   // Handle keyboard 'event'
   if (IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleKey", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   char tmp[10] = {0};
   UInt_t keysym = 0;

   if (fGLDevice == -1)
      gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   else
      keysym = event->fCode;
   fRnrCtx->SetEventKeySym(keysym);

   Bool_t redraw = kFALSE;
   if (fCurrentOvlElm && fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event))
   {
      redraw = kTRUE;
   }
   else
   {
      switch (keysym)
      {
         case kKey_R:
         case kKey_r:
            SetStyle(TGLRnrCtx::kFill);
            if (fClearColor == 0) {
               fClearColor = 1; // Black
               RefreshPadEditor(this);
            }
            redraw = kTRUE;
            break;
         case kKey_W:
         case kKey_w:
            SetStyle(TGLRnrCtx::kWireFrame);
            if (fClearColor == 0) {
               fClearColor = 1; // Black
               RefreshPadEditor(this);
            }
            redraw = kTRUE;
            break;
         case kKey_T:
         case kKey_t:
            SetStyle(TGLRnrCtx::kOutline);
            if (fClearColor == 1) {
               fClearColor = 0; // White
               RefreshPadEditor(this);
            }
            redraw = kTRUE;
            break;

            // Camera
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
         case kKey_Home:
            ResetCurrentCamera();
            redraw = kTRUE;
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
         default:;
      } // switch
   }

   if (redraw) {
      if (fGLDevice != -1)
         gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
      RequestDraw();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleMotion(Event_t * event)
{
   // Handle mouse motion 'event'
   if (IsLocked()) {
      if (gDebug>3) {
         Info("TGLViewer::HandleMotion", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   assert (event); // was if event==0 return

   Bool_t processed = kFALSE, changed = kFALSE;
   Short_t lod = TGLRnrCtx::kLODMed;

   // Camera interface requires GL coords - Y inverted
   Int_t xDelta = event->fX - fLastPos.fX;
   Int_t yDelta = event->fY - fLastPos.fY;

   if (fAction == kDragNone)
   {
      changed = RequestOverlaySelect(event->fX, event->fY);
      if (fCurrentOvlElm)
         processed = fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event);
      lod = TGLRnrCtx::kLODHigh;
   } else if (fAction == kDragCameraRotate) {
      processed = CurrentCamera().Rotate(xDelta, -yDelta);
   } else if (fAction == kDragCameraTruck) {
      processed = CurrentCamera().Truck(event->fX, fViewport.Y() - event->fY,
                                        xDelta, -yDelta);
   } else if (fAction == kDragCameraDolly) {
      processed = CurrentCamera().Dolly(xDelta, event->fState & kKeyControlMask,
                                        event->fState & kKeyShiftMask);
   } else if (fAction == kDragOverlay) {
      processed = fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event);
   }

   fLastPos.fX = event->fX;
   fLastPos.fY = event->fY;

   if (processed || changed) {
      if (fGLDevice != -1) {
         gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
         gVirtualX->SetDrawMode(TVirtualX::kCopy);
      }

      RequestDraw(lod);
   }

   return processed;
}

//______________________________________________________________________________
Bool_t TGLViewer::HandleExpose(Event_t * event)
{
   // Handle window expose 'event' - show
   if (event->fCount != 0) return kTRUE;

   if (IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleExpose", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return kFALSE;
   }

   fRedrawTimer->RequestDraw(20, TGLRnrCtx::kLODHigh);
   return kTRUE;
}

//______________________________________________________________________________
void TGLViewer::Repaint()
{
   // Handle window expose 'event' - show
   if (IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleExpose", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return;
   }

   fRedrawTimer->RequestDraw(20, TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
TClass* TGLViewer::FindDirectRendererClass(TClass* cls)
{
   TString rnr( cls->GetName() );
   rnr += "GL";
   TClass* c = TClass::GetClass(rnr);
   if (c != 0)
      return c;

   TList* bases = cls->GetListOfBases();
   if (bases == 0 || bases->IsEmpty())
      return 0;

   TIter  next_base(bases);
   TBaseClass* bc;
   while ((bc = (TBaseClass*) next_base()) != 0) {
      cls = bc->GetClassPointer();
      if ((c = FindDirectRendererClass(cls)) != 0) {
         return c;
      }
   }
   return 0;
}

//______________________________________________________________________________
TGLLogicalShape* TGLViewer::AttemptDirectRenderer(TObject* id)
{
   TClass* isa = id->IsA();
   std::map<TClass*, TClass*>::iterator i = fDirectRendererMap.find(isa);
   TClass* cls;
   if (i != fDirectRendererMap.end()) {
      cls = i->second;
   } else {
      cls = FindDirectRendererClass(isa);
      fDirectRendererMap[isa] = cls;
   }
   TGLObject* rnr = 0;
   if (cls != 0) {
      rnr = reinterpret_cast<TGLObject*>(cls->New());
      if (rnr) {
         if (rnr->SetModel(id) == false) {
            Warning("TGLViewer::AttemptDirectRenderer", "failed initializing direct rendering.");
            delete rnr;
            return 0;
         }
         rnr->SetBBox();
         fScene.AdoptLogical(*rnr);
      }
   }
   return rnr;
}

//______________________________________________________________________________
void TGLViewer::PrintObjects()
{
   // Pass viewer for print capture by TGLOutput.

   TGLOutput::Capture(*this);
}
