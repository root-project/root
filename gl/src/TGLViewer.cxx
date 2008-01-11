// @(#)root/gl:$Id$
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

#include "TGLScenePad.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLObject.h"
#include "TGLStopwatch.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGLOutput.h"

#include "TVirtualPad.h" // Remove when pad removed - use signal
#include "TAtt3D.h"      // Remove when PadPaint delegated to PadScene.
#include "TVirtualX.h"

#include "TMath.h"
#include "TColor.h"
#include "TError.h"
#include "TClass.h"
#include "TROOT.h"

// For event type translation ExecuteEvent
#include "Buttons.h"
#include "GuiTypes.h"

#include "TVirtualGL.h"

#include "TGLWidget.h"
#include "TGLViewerEditor.h"

#include "KeySymbols.h"
#include "TContextMenu.h"


//______________________________________________________________________
// TGLViewer
//
// Base GL viewer object - used by both standalone and embedded (in pad)
// GL. Contains core viewer objects :
//
// GL scene - collection of main drawn objects - see TGLStdScene
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
   fPerspectiveCameraXOZ(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // XOZ floor
   fPerspectiveCameraYOZ(TGLVector3( 0.0,-1.0, 0.0), TGLVector3(1.0, 0.0, 0.0)), // YOZ floor
   fPerspectiveCameraXOY(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // XOY floor
   fOrthoXOYCamera(TGLVector3( 0.0, 0.0, 1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down Z axis, X horz, Y vert
   fOrthoXOZCamera(TGLVector3( 0.0,-1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking down Y axis, X horz, Z vert
   fOrthoZOYCamera(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down X axis, Z horz, Y vert
   fCurrentCamera(&fPerspectiveCameraXOZ),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fPushAction(kPushStd), fAction(kDragNone), fLastPos(0,0), fActiveButtonID(0),
   fRedrawTimer(0),
   fClearColor(1),
   fAxesType(TGLUtil::kAxesNone),
   fAxesDepthTest(kTRUE),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fDrawCameraCenter(kFALSE),
   fCameraMarkup(0),
   fInitGL(kFALSE),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fIsPrinting(kFALSE),
   fGLWindow(0),
   fGLDevice(-1),
   fGLCtxId(0),
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
   fPerspectiveCameraXOZ(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // XOZ floor
   fPerspectiveCameraYOZ(TGLVector3( 0.0,-1.0, 0.0), TGLVector3(1.0, 0.0, 0.0)), // YOZ floor
   fPerspectiveCameraXOY(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // XOY floor
   fOrthoXOYCamera(TGLVector3( 0.0, 0.0, 1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down Z axis, X horz, Y vert
   fOrthoXOZCamera(TGLVector3( 0.0,-1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking down Y axis, X horz, Z vert
   fOrthoZOYCamera(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down X axis, Z horz, Y vert
   fCurrentCamera(&fPerspectiveCameraXOZ),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fPushAction(kPushStd), fAction(kDragNone), fLastPos(0,0), fActiveButtonID(0),
   fRedrawTimer(0),
   fClearColor(1),
   fAxesType(TGLUtil::kAxesNone),
   fAxesDepthTest(kTRUE),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fDrawCameraCenter(kFALSE),
   fCameraMarkup(0),
   fInitGL(kFALSE),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fIsPrinting(kFALSE),
   fGLWindow(0),
   fGLDevice(fPad->GetGLDevice()),
   fGLCtxId(0),
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
   if (fPad)
      fPad->ReleaseViewer3D();
   if (fGLDevice != -1)
      fGLCtxId->Release(0);
}


//______________________________________________________________________________
void TGLViewer::PadPaint(TVirtualPad* pad)
{
   // Entry point for updating viewer contents via VirtualViewer3D
   // interface.
   // We search and forward the request to appropriate TGLScenePad.
   // If it is not found we create a new TGLScenePad so this can
   // potentially also be used for registration of new pads.

   TGLScenePad* scenepad = 0;
   for (SceneInfoList_i si = fScenes.begin(); si != fScenes.end(); ++si)
   {
      scenepad = dynamic_cast<TGLScenePad*>((*si)->GetScene());
      if (scenepad && scenepad->GetPad() == pad)
         break;
      scenepad = 0;
   }
   if (scenepad == 0)
   {
      scenepad = new TGLScenePad(pad);
      AddScene(scenepad);
   }

   scenepad->PadPaintFromViewer(this);

   PostSceneBuildSetup(fResetCamerasOnNextUpdate || fResetCamerasOnUpdate);
   fResetCamerasOnNextUpdate = kFALSE;

   RequestDraw();
}


/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::UpdateScene()
{
   // Force update of pad-scenes. Eventually this could be generalized
   // to all scene-types via a virtual function in TGLSceneBase.

   // Cancel any pending redraw timer.
   fRedrawTimer->Stop();

   for (SceneInfoList_i si = fScenes.begin(); si != fScenes.end(); ++si)
   {
      TGLScenePad* scenepad = dynamic_cast<TGLScenePad*>((*si)->GetScene());
      if (scenepad)
         scenepad->PadPaintFromViewer(this);
   }

   PostSceneBuildSetup(fResetCamerasOnNextUpdate || fResetCamerasOnUpdate);
   fResetCamerasOnNextUpdate = kFALSE;

   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::ResetCurrentCamera()
{
   // Resets position/rotation of current camera to default values.

   MergeSceneBBoxes(fOverallBoundingBox);
   CurrentCamera().Setup(fOverallBoundingBox, kTRUE);
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
   // Perform post scene-build setup.

   MergeSceneBBoxes(fOverallBoundingBox);
   SetupCameras(resetCameras);

   // Set default reference to center
   fReferencePos.Set(fOverallBoundingBox.Center());
   RefreshPadEditor(this);
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
   if (fVisScenes.size() > 1)
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

   if (CurrentCamera().UpdateInterest(kFALSE)) {
      // Reset major view-dependant cache.
      ResetSceneInfos();
      redrawReq = kTRUE;
   }
   if (fLOD != TGLRnrCtx::kLODHigh) {
      // Request final draw pass.
      redrawReq = kTRUE;
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

   Bool_t disabled = kFALSE;
   if (fReferenceOn)
   {
      glDisable(GL_DEPTH_TEST);
      TGLUtil::DrawReferenceMarker(*fCamera, fReferencePos);
      disabled = kTRUE;
   }
   if (fDrawCameraCenter)
   {
      glDisable(GL_DEPTH_TEST);
      Float_t radius = fCamera->ViewportDeltaToWorld(TGLVertex3(fCamera->GetCenterVec()), 3, 3).Mag();
      const Float_t rgba[4] = { 0, 1, 1, 1.0 };
      TGLUtil::DrawSphere(fCamera->GetCenterVec(), radius, rgba);
      disabled = kTRUE;
   }
   if(fAxesDepthTest && disabled)
   {
      glEnable(GL_DEPTH_TEST);
      disabled = kFALSE;
   }
   else if (fAxesDepthTest == kFALSE && disabled == kFALSE)
   {
      glDisable(GL_DEPTH_TEST);
      disabled = kTRUE;
   }
   TGLUtil::DrawSimpleAxes(*fCamera, fOverallBoundingBox, fAxesType);
   if(disabled)
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

   if (gDebug>2) {
      Info("TGLViewer::SetViewport", "updated - corner %d,%d dimensions %d,%d", x, y, width, height);
   }
}


/**************************************************************************/
// Camera methods
/**************************************************************************/

//______________________________________________________________________________
TGLCamera& TGLViewer::RefCamera(ECameraType cameraType)
{
   // Return camera reference by type.

   // TODO: Move these into a vector!
   switch(cameraType) {
      case kCameraPerspXOZ:
         return fPerspectiveCameraXOZ;
      case kCameraPerspYOZ:
         return fPerspectiveCameraYOZ;
      case kCameraPerspXOY:
         return fPerspectiveCameraXOY;
      case kCameraOrthoXOY:
         return fOrthoXOYCamera;
      case kCameraOrthoXOZ:
         return fOrthoXOZCamera;
      case kCameraOrthoZOY:
         return fOrthoZOYCamera;
      default:
         Error("TGLViewer::SetCurrentCamera", "invalid camera type");
         return *fCurrentCamera;
   }
}

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
      case kCameraPerspXOZ: {
         fCurrentCamera = &fPerspectiveCameraXOZ;
         break;
      }
      case kCameraPerspYOZ: {
         fCurrentCamera = &fPerspectiveCameraYOZ;
         break;
      }
      case kCameraPerspXOY: {
         fCurrentCamera = &fPerspectiveCameraXOY;
         break;
      }
      case kCameraOrthoXOY: {
         fCurrentCamera = &fOrthoXOYCamera;
         break;
      }
      case kCameraOrthoXOZ: {
         fCurrentCamera = &fOrthoXOZCamera;
         break;
      }
      case kCameraOrthoZOY: {
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
                                     Double_t zoom, Double_t dolly,
                                     Double_t center[3],
                                     Double_t hRotate, Double_t vRotate)
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
      case kCameraOrthoXOY: {
         fOrthoXOYCamera.Configure(zoom, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fOrthoXOYCamera) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case kCameraOrthoXOZ: {
         fOrthoXOZCamera.Configure(zoom, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fOrthoXOZCamera) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case kCameraOrthoZOY: {
         fOrthoZOYCamera.Configure(zoom, dolly, center, hRotate, vRotate);
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
      case kCameraPerspXOZ: {
         fPerspectiveCameraXOZ.Configure(fov, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fPerspectiveCameraXOZ) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case kCameraPerspYOZ: {
         fPerspectiveCameraYOZ.Configure(fov, dolly, center, hRotate, vRotate);
         if (fCurrentCamera == &fPerspectiveCameraYOZ) {
            RequestDraw(TGLRnrCtx::kLODHigh);
         }
         break;
      }
      case kCameraPerspXOY: {
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
void TGLViewer::GetGuideState(Int_t & axesType, Bool_t & axesDepthTest, Bool_t & referenceOn, Double_t referencePos[3]) const
{
   // Fetch the state of guides (axes & reference markers) into arguments

   axesType = fAxesType;
   axesDepthTest = fAxesDepthTest;

   referenceOn = fReferenceOn;
   referencePos[0] = fReferencePos.X();
   referencePos[1] = fReferencePos.Y();
   referencePos[2] = fReferencePos.Z();
}

//______________________________________________________________________________
void TGLViewer::SetGuideState(Int_t axesType, Bool_t axesDepthTest, Bool_t referenceOn, const Double_t referencePos[3])
{
   // Set the state of guides (axes & reference markers) from arguments.

   fAxesType    = axesType;
   fAxesDepthTest = axesDepthTest;
   fReferenceOn = referenceOn;
   fReferencePos.Set(referencePos[0], referencePos[1], referencePos[2]);
   if (fGLDevice != -1)
      gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
   RequestDraw();
}

//______________________________________________________________________________
void TGLViewer::SetDrawCameraCenter(Bool_t x)
{
   // Draw camera look at and rotation point.

   fDrawCameraCenter = x;
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
   // Emit signal indicating selection has changed.

   Emit("SelectionChanged()");
}

//______________________________________________________________________________
void TGLViewer::OverlayDragFinished()
{
   // Emit signal indicating that an overlay drag has finished.

   Emit("OverlayDragFinished()");
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
   // and terminate any interaction in viewer.

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
   // Handle mouse button 'event'.

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

      if (fPushAction == kPushCamCenter)
      {
         fPushAction = kPushStd;
         RequestSelect(event->fX, event->fY);
         if (fSelRec.GetN() > 0)
         {
            TGLVector3 v(event->fX, event->fY, 0.5*fSelRec.GetMinZ());
            fCurrentCamera->WindowToViewport(v);
            v = fCurrentCamera->ViewportToWorld(v);
            fCurrentCamera->SetExternalCenter(kTRUE);
            fCurrentCamera->SetCenterVec(v.X(), v.Y(), v.Z());
            RequestDraw();
         }
         RefreshPadEditor(this);
         return kTRUE;
      }

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
                     handled = kTRUE;
                  } else {
                     SelectionChanged(); // Just notify clients.
                  }
               } else if (event->fState & kKeyControlMask) {
                  RequestSelect(event->fX, event->fY, kTRUE);
                  if (fSecSelRec.GetPhysShape() != 0) {
                     TGLLogicalShape& lshape = const_cast<TGLLogicalShape&>
                        (*fSecSelRec.GetPhysShape()->GetLogical());
                     lshape.ProcessSelection(*fRnrCtx, fSecSelRec);
                     handled = kTRUE;
                  }
               }
               if ( ! handled) {
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
         OverlayDragFinished();
         if (RequestOverlaySelect(event->fX, event->fY))
            RequestDraw();
      }

      // TODO: Check on Linux - on Win32 only see button release events
      // for mouse wheel
      switch(event->fCode) {
         // Buttons 4/5 are mouse wheel
         // Note: Modifiers (ctrl/shift) disabled as fState doesn't seem to
         // have correct modifier flags with mouse wheel under Windows.
         case kButton5: {
            // Zoom out (adjust camera FOV)
            if (CurrentCamera().Zoom(+50, kFALSE, kFALSE)) { //TODO : val static const somewhere
               RequestDraw();
            }
            break;
         }
         case kButton4: {
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
   // Handle mouse double click 'event'.

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
   // Handle configure notify 'event' - a window resize/movement.

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
   // Handle keyboard 'event'.

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
               UpdateScene();
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
   // Handle mouse motion 'event'.

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
   Int_t  xDelta = event->fX - fLastPos.fX;
   Int_t  yDelta = event->fY - fLastPos.fY;
   Bool_t mod1   = event->fState & kKeyControlMask;
   Bool_t mod2   = event->fState & kKeyShiftMask;

   if (fAction == kDragNone)
   {
      changed = RequestOverlaySelect(event->fX, event->fY);
      if (fCurrentOvlElm)
         processed = fCurrentOvlElm->Handle(*fRnrCtx, fOvlSelRec, event);
      lod = TGLRnrCtx::kLODHigh;
   } else if (fAction == kDragCameraRotate) {
      processed = CurrentCamera().Rotate(xDelta, -yDelta, mod1, mod2);
   } else if (fAction == kDragCameraTruck) {
      processed = CurrentCamera().Truck(xDelta, -yDelta, mod1, mod2);
   } else if (fAction == kDragCameraDolly) {
      processed = CurrentCamera().Dolly(xDelta, mod1, mod2);
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
   // Handle window expose 'event' - show.

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
   // Handle window expose 'event' - show.

   if (IsLocked()) {
      if (gDebug > 0) {
         Info("TGLViewer::HandleExpose", "ignored - viewer is %s", LockName(CurrentLock()));
      }
      return;
   }

   fRedrawTimer->RequestDraw(20, TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
void TGLViewer::PrintObjects()
{
   // Pass viewer for print capture by TGLOutput.

   TGLOutput::Capture(*this);
}
