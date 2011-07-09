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
#include "TGLManipSet.h"
#include "TGLCameraOverlay.h"
#include "TGLAutoRotator.h"

#include "TGLScenePad.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLObject.h"
#include "TGLStopwatch.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGLOutput.h"

#include "TVirtualPad.h" // Remove when pad removed - use signal
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
#include "TGLFBO.h"
#include "TGLViewerEditor.h"
#include "TGedEditor.h"
#include "TGLPShapeObj.h"

#include "KeySymbols.h"
#include "TContextMenu.h"
#include "TImage.h"

#include <stdexcept>

#ifndef GL_BGRA
#define GL_BGRA GL_BGRA_EXT
#endif

//==============================================================================
// TGLViewer
//==============================================================================

//______________________________________________________________________
//
// Base GL viewer object - used by both standalone and embedded (in pad)
// GL. Contains core viewer objects :
//
// GL scene - collection of main drawn objects - see TGLStdScene
// Cameras (fXyzzCamera) - ortho and perspective cameras - see TGLCamera
// Clipping (fClipXyzz) - collection of clip objects - see TGLClip
// Manipulators (fXyzzManip) - collection of manipulators - see TGLManip
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
// Also, the TGLWidget needs to be created externally. It is not owned
// by the viewer.
//
// For embedded (pad) GL this viewer is created directly by plugin
// manager. For standalone the derived TGLSAViewer is.
//

ClassImp(TGLViewer);

TGLColorSet TGLViewer::fgDefaultColorSet;
Bool_t      TGLViewer::fgUseDefaultColorSetForNewViewers = kFALSE;

//______________________________________________________________________________
TGLViewer::TGLViewer(TVirtualPad * pad, Int_t x, Int_t y,
                     Int_t width, Int_t height) :
   fPad(pad),
   fContextMenu(0),
   fPerspectiveCameraXOZ(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // XOZ floor
   fPerspectiveCameraYOZ(TGLVector3( 0.0,-1.0, 0.0), TGLVector3(1.0, 0.0, 0.0)), // YOZ floor
   fPerspectiveCameraXOY(TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // XOY floor
   fOrthoXOYCamera (TGLOrthoCamera::kXOY,  TGLVector3( 0.0, 0.0, 1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down  Z axis,  X horz, Y vert
   fOrthoXOZCamera (TGLOrthoCamera::kXOZ,  TGLVector3( 0.0,-1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking along Y axis,  X horz, Z vert
   fOrthoZOYCamera (TGLOrthoCamera::kZOY,  TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking along X axis,  Z horz, Y vert
   fOrthoXnOYCamera(TGLOrthoCamera::kXnOY, TGLVector3( 0.0, 0.0,-1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking along Z axis, -X horz, Y vert
   fOrthoXnOZCamera(TGLOrthoCamera::kXnOZ, TGLVector3( 0.0, 1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking down  Y axis, -X horz, Z vert
   fOrthoZnOYCamera(TGLOrthoCamera::kZnOY, TGLVector3( 1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down  X axis, -Z horz, Y vert
   fCurrentCamera(&fPerspectiveCameraXOZ),
   fAutoRotator(0),

   fStereo               (kFALSE),
   fStereoZeroParallax   (0.03f),
   fStereoEyeOffsetFac   (1.0f),
   fStereoFrustumAsymFac (1.0f),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fEventHandler(0),
   fGedEditor(0),
   fPShapeWrap(0),
   fPushAction(kPushStd), fDragAction(kDragNone),
   fRedrawTimer(0),
   fMaxSceneDrawTimeHQ(5000),
   fMaxSceneDrawTimeLQ(100),
   fPointScale (1), fLineScale(1), fSmoothPoints(kFALSE), fSmoothLines(kFALSE),
   fAxesType(TGLUtil::kAxesNone),
   fAxesDepthTest(kTRUE),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fDrawCameraCenter(kFALSE),
   fCameraOverlay(0),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fIsPrinting(kFALSE),
   fPictureFileName("viewer.jpg"),
   fFader(0),
   fGLWidget(0),
   fGLDevice(-1),
   fGLCtxId(0),
   fIgnoreSizesOnUpdate(kFALSE),
   fResetCamerasOnUpdate(kTRUE),
   fResetCamerasOnNextUpdate(kFALSE)
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
   fOrthoXOYCamera (TGLOrthoCamera::kXOY,  TGLVector3( 0.0, 0.0, 1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down  Z axis,  X horz, Y vert
   fOrthoXOZCamera (TGLOrthoCamera::kXOZ,  TGLVector3( 0.0,-1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking along Y axis,  X horz, Z vert
   fOrthoZOYCamera (TGLOrthoCamera::kZOY,  TGLVector3(-1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking along X axis,  Z horz, Y vert
   fOrthoXnOYCamera(TGLOrthoCamera::kXnOY, TGLVector3( 0.0, 0.0,-1.0), TGLVector3(0.0, 1.0, 0.0)), // Looking along Z axis, -X horz, Y vert
   fOrthoXnOZCamera(TGLOrthoCamera::kXnOZ, TGLVector3( 0.0, 1.0, 0.0), TGLVector3(0.0, 0.0, 1.0)), // Looking down  Y axis, -X horz, Z vert
   fOrthoZnOYCamera(TGLOrthoCamera::kZnOY, TGLVector3( 1.0, 0.0, 0.0), TGLVector3(0.0, 1.0, 0.0)), // Looking down  X axis, -Z horz, Y vert
   fCurrentCamera(&fPerspectiveCameraXOZ),
   fAutoRotator(0),

   fStereo               (kFALSE),
   fStereoZeroParallax   (0.03f),
   fStereoEyeOffsetFac   (1.0f),
   fStereoFrustumAsymFac (1.0f),

   fLightSet          (0),
   fClipSet           (0),
   fSelectedPShapeRef (0),
   fCurrentOvlElm     (0),

   fEventHandler(0),
   fGedEditor(0),
   fPShapeWrap(0),
   fPushAction(kPushStd), fDragAction(kDragNone),
   fRedrawTimer(0),
   fMaxSceneDrawTimeHQ(5000),
   fMaxSceneDrawTimeLQ(100),
   fPointScale (1), fLineScale(1), fSmoothPoints(kFALSE), fSmoothLines(kFALSE),
   fAxesType(TGLUtil::kAxesNone),
   fAxesDepthTest(kTRUE),
   fReferenceOn(kFALSE),
   fReferencePos(0.0, 0.0, 0.0),
   fDrawCameraCenter(kFALSE),
   fCameraOverlay(0),
   fSmartRefresh(kFALSE),
   fDebugMode(kFALSE),
   fIsPrinting(kFALSE),
   fPictureFileName("viewer.jpg"),
   fFader(0),
   fGLWidget(0),
   fGLDevice(fPad->GetGLDevice()),
   fGLCtxId(0),
   fIgnoreSizesOnUpdate(kFALSE),
   fResetCamerasOnUpdate(kTRUE),
   fResetCamerasOnNextUpdate(kFALSE)
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
   fClipSet  = new TGLClipSet;
   AddOverlayElement(fClipSet);

   fSelectedPShapeRef = new TGLManipSet;
   fSelectedPShapeRef->SetDrawBBox(kTRUE);
   AddOverlayElement(fSelectedPShapeRef);

   fPShapeWrap = new TGLPShapeObj(0, this);

   fLightColorSet.StdLightBackground();
   if (fgUseDefaultColorSetForNewViewers) {
      fRnrCtx->ChangeBaseColorSet(&fgDefaultColorSet);
   } else {
      if (fPad) {
         fRnrCtx->ChangeBaseColorSet(&fLightColorSet);
         fLightColorSet.Background().SetColor(fPad->GetFillColor());
         fLightColorSet.Foreground().SetColor(fPad->GetLineColor());
      } else {
         fRnrCtx->ChangeBaseColorSet(&fDarkColorSet);
      }
   }

   fCameraOverlay = new TGLCameraOverlay(kFALSE, kFALSE);
   AddOverlayElement(fCameraOverlay);

   fRedrawTimer = new TGLRedrawTimer(*this);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
   // Destroy viewer object.

   delete fAutoRotator;

   delete fLightSet;
   // fClipSet, fSelectedPShapeRef and fCameraOverlay deleted via overlay.

   delete fContextMenu;
   delete fRedrawTimer;

   if (fEventHandler) {
      if (fGLWidget)
         fGLWidget->SetEventHandler(0);
      delete fEventHandler;
   }

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
void TGLViewer::UpdateScene(Bool_t redraw)
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

   if (redraw)
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
      fOrthoXnOYCamera.Setup(box, reset);
      fOrthoXnOZCamera.Setup(box, reset);
      fOrthoZnOYCamera.Setup(box, reset);
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
   // Initialise GL state.

   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glClearColor(0.f, 0.f, 0.f, 0.f);
   glClearDepth(1.0);
   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glMaterialf(GL_BACK, GL_SHININESS, 0.0);
   glPolygonMode(GL_FRONT, GL_FILL);
   glDisable(GL_BLEND);

   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodelAmb);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

   glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

   TGLUtil::CheckError("TGLViewer::InitGL");
}

//______________________________________________________________________________
void TGLViewer::RequestDraw(Short_t LODInput)
{
   // Post request for redraw of viewer at level of detail 'LOD'
   // Request is directed via cross thread gVirtualGL object.

   fRedrawTimer->Stop();
   // Ignore request if GL window or context not yet availible or shown.
   if ((!fGLWidget && fGLDevice == -1) || (fGLWidget && !fGLWidget->IsMapped()))
   {
      return;
   }

   // Take scene draw lock - to be revisited
   if ( ! TakeLock(kDrawLock)) {
      // If taking drawlock fails the previous draw is still in progress
      // set timer to do this one later
      if (gDebug>3) {
         Info("TGLViewer::RequestDraw", "viewer locked - requesting another draw.");
      }
      fRedrawTimer->RequestDraw(100, LODInput);
      return;
   }
   fLOD = LODInput;

   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoDraw()", (ULong_t)this));
   else
      DoDraw();
}

//______________________________________________________________________________
void TGLViewer::SetupClipObject()
{
   // Setup clip-object. Protected virtual method.

   if (GetClipAutoUpdate())
   {
      fClipSet->SetupCurrentClip(fOverallBoundingBox);
   }
   else
   {
      fClipSet->SetupCurrentClipIfInvalid(fOverallBoundingBox);
   }
}
//______________________________________________________________________________
void TGLViewer::PreRender()
{
   // Initialize objects that influence rendering.
   // Called before every render.

   fCamera = fCurrentCamera;
   fClip   = fClipSet->GetCurrentClip();
   if (fGLDevice != -1)
   {
      fRnrCtx->SetGLCtxIdentity(fGLCtxId);
      fGLCtxId->DeleteGLResources();
   }

   TGLUtil::SetPointSizeScale(fPointScale * fRnrCtx->GetRenderScale());
   TGLUtil::SetLineWidthScale(fLineScale  * fRnrCtx->GetRenderScale());

   if (fSmoothPoints) glEnable(GL_POINT_SMOOTH); else glDisable(GL_POINT_SMOOTH);
   if (fSmoothLines)  glEnable(GL_LINE_SMOOTH);  else glDisable(GL_LINE_SMOOTH);
   if (fSmoothPoints || fSmoothLines)
   {
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glEnable(GL_BLEND);
   }
   else
   {
      glDisable(GL_BLEND);
   }

   TGLViewerBase::PreRender();

   // Setup lighting
   fLightSet->StdSetupLights(fOverallBoundingBox, *fCamera, fDebugMode);
}

//______________________________________________________________________________
void TGLViewer::PostRender()
{
   // Restore state set in PreRender().
   // Called after every render.

   TGLViewerBase::PostRender();

   TGLUtil::SetPointSizeScale(1);
   TGLUtil::SetLineWidthScale(1);
}

//______________________________________________________________________________
void TGLViewer::DoDraw(Bool_t swap_buffers)
{
   // Draw out the viewer.

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

   TUnlocker ulck(this);

   if (fGLDevice == -1 && (fViewport.Width() <= 1 || fViewport.Height() <= 1)) {
      if (gDebug > 2) {
	 Info("TGLViewer::DoDraw()", "zero surface area, draw skipped.");
      }
      return;
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

   // Setup scene draw time
   fRnrCtx->SetRenderTimeOut(fLOD == TGLRnrCtx::kLODHigh ?
                             fMaxSceneDrawTimeHQ :
                             fMaxSceneDrawTimeLQ);

   if (fStereo && fCurrentCamera->IsPerspective() && !fRnrCtx->GetGrabImage() &&
       !fIsPrinting)
   {
      DoDrawStereo(swap_buffers);
   }
   else
   {
      DoDrawMono(swap_buffers);
   }

   ReleaseLock(kDrawLock);

   if (gDebug>2) {
      Info("TGLViewer::DoDraw()", "Took %f msec", timer.End());
   }

   // Check if further redraws are needed and schedule them.

   if (CurrentCamera().UpdateInterest(kFALSE)) {
      // Reset major view-dependant cache.
      ResetSceneInfos();
      fRedrawTimer->RequestDraw(0, fLOD);
   }

   if (fLOD != TGLRnrCtx::kLODHigh &&
       (fDragAction < kDragCameraRotate || fDragAction > kDragCameraDolly))
   {
      // Request final draw pass.
      fRedrawTimer->RequestDraw(100, TGLRnrCtx::kLODHigh);
   }
}

//______________________________________________________________________________
void TGLViewer::DoDrawMono(Bool_t swap_buffers)
{
   // Draw out in monoscopic mode.

   MakeCurrent();

   if (!fIsPrinting) PreDraw();
   PreRender();

   fRnrCtx->StartStopwatch();
   if (fFader < 1)
   {
      RenderNonSelected();
      RenderSelected();
      DrawGuides();
      RenderOverlay(TGLOverlayElement::kAllVisible, kFALSE);

      glClear(GL_DEPTH_BUFFER_BIT);
      fRnrCtx->SetHighlight(kTRUE);
      RenderSelected();
      fRnrCtx->SetHighlight(kFALSE);
      glClear(GL_DEPTH_BUFFER_BIT);
      DrawDebugInfo();
   }
   fRnrCtx->StopStopwatch();

   PostRender();

   if (fFader > 0)
   {
      FadeView(fFader);
   }

   PostDraw();

   if (swap_buffers)
   {
      SwapBuffers();
   }
}

//______________________________________________________________________________
void TGLViewer::DoDrawStereo(Bool_t swap_buffers)
{
   // Draw out in stereoscopic mode.

   TGLPerspectiveCamera &c = *dynamic_cast<TGLPerspectiveCamera*>(fCurrentCamera);

   Float_t gl_near, gl_far, zero_p_dist;
   Float_t h_half, w_half;
   Float_t x_len_at_zero_parallax;
   Float_t stereo_offset;
   Float_t frustum_asym;

   MakeCurrent();

   // Draw left
   glDrawBuffer(GL_BACK_LEFT);
   PreDraw();
   PreRender();

   gl_near = c.GetNearClip();
   gl_far  = c.GetFarClip();
   zero_p_dist = gl_near + fStereoZeroParallax*(gl_far-gl_near);

   h_half = TMath::Tan(0.5*TMath::DegToRad()*c.GetFOV()) * gl_near;
   w_half = h_half * fViewport.Aspect();

   x_len_at_zero_parallax = 2.0f * w_half * zero_p_dist / gl_near;
   stereo_offset = 0.035f * x_len_at_zero_parallax * fStereoEyeOffsetFac;

   frustum_asym = stereo_offset * gl_near / zero_p_dist * fStereoFrustumAsymFac;

   TGLMatrix  abs_trans(c.RefCamBase());
   abs_trans *= c.RefCamTrans();
   TGLVector3 left_vec = abs_trans.GetBaseVec(2);

   glTranslatef(stereo_offset*left_vec[0], stereo_offset*left_vec[1], stereo_offset*left_vec[2]);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glFrustum(-w_half + frustum_asym, w_half + frustum_asym,
             -h_half, h_half, gl_near, gl_far);
   glMatrixMode(GL_MODELVIEW);

   fRnrCtx->StartStopwatch();
   if (fFader < 1)
   {
      RenderNonSelected();
      RenderSelected();
      DrawGuides();
      RenderOverlay(TGLOverlayElement::kAllVisible, kFALSE);

      glClear(GL_DEPTH_BUFFER_BIT);
      fRnrCtx->SetHighlight(kTRUE);
      RenderSelected();
      fRnrCtx->SetHighlight(kFALSE);
      glClear(GL_DEPTH_BUFFER_BIT);
      DrawDebugInfo();
   }
   fRnrCtx->StopStopwatch();

   PostRender();

   if (fFader > 0)
   {
      FadeView(fFader);
   }
   PostDraw();

   // Draw right
   glDrawBuffer(GL_BACK_RIGHT);
   PreDraw();
   PreRender();

   glTranslatef(-stereo_offset*left_vec[0], -stereo_offset*left_vec[1], -stereo_offset*left_vec[2]);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glFrustum(-w_half - frustum_asym, w_half - frustum_asym,
             -h_half, h_half, gl_near, gl_far);
   glMatrixMode(GL_MODELVIEW);

   fRnrCtx->StartStopwatch();
   if (fFader < 1)
   {
      RenderNonSelected();
      RenderSelected();
      DrawGuides();
      RenderOverlay(TGLOverlayElement::kAllVisible, kFALSE);

      glClear(GL_DEPTH_BUFFER_BIT);
      fRnrCtx->SetHighlight(kTRUE);
      RenderSelected();
      fRnrCtx->SetHighlight(kFALSE);
      glClear(GL_DEPTH_BUFFER_BIT);
      DrawDebugInfo();
   }
   fRnrCtx->StopStopwatch();

   PostRender();

   if (fFader > 0)
   {
      FadeView(fFader);
   }
   PostDraw();

   // End
   if (swap_buffers)
   {
      SwapBuffers();
   }

   glDrawBuffer(GL_BACK);
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePicture()
{
   // Save current image using the defualt file name which can be set
   // via SetPictureFileName() and defaults to "viewer.jpg".
   // Really useful for the files ending with 'gif+'.

   return SavePicture(fPictureFileName);
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePicture(const TString &fileName)
{
   // Save current image in various formats (gif, gif+, jpg, png, eps, pdf).
   // 'gif+' will append image to an existng file (animated gif).
   // 'eps' and 'pdf' do not fully support transparency and texturing.
   // The viewer window most be fully contained within the desktop but
   // can be covered by other windows.
   // Returns false if something obvious goes wrong, true otherwise.
   //
   // The mage is saved using a frame-buffer object if the GL implementation
   // claims to support it -- this claim is not always true, especially when
   // running over ssh with drastically different GL implementations on the
   // client and server sides. Set this in .rootrc to enforce creation of
   // pictures using the back-buffer:
   //   OpenGL.SavePicturesViaFBO: off

   if (fileName.EndsWith(".eps"))
   {
      return TGLOutput::Capture(*this, TGLOutput::kEPS_BSP, fileName.Data());
   }
   else if (fileName.EndsWith(".pdf"))
   {
      return TGLOutput::Capture(*this, TGLOutput::kPDF_BSP, fileName.Data());
   }
   else
   {
      if (GLEW_EXT_framebuffer_object && gEnv->GetValue("OpenGL.SavePicturesViaFBO", 1))
      {
         return SavePictureUsingFBO(fileName, fViewport.Width(), fViewport.Height(), kFALSE);
      }
      else
      {
         return SavePictureUsingBB(fileName);
      }
   }
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePictureUsingBB(const TString &fileName)
{
   // Save current image in various formats (gif, gif+, jpg, png).
   // 'gif+' will append image to an existng file (animated gif).
   // Back-Buffer is used for capturing of the image.
   // The viewer window most be fully contained within the desktop but
   // can be covered by other windows.
   // Returns false if something obvious goes wrong, true otherwise.

   static const TString eh("TGLViewer::SavePictureUsingBB");

   if (! fileName.EndsWith(".gif") && ! fileName.Contains(".gif+") &&
       ! fileName.EndsWith(".jpg") && ! fileName.EndsWith(".png"))
   {
      Warning(eh, "file %s cannot be saved with this extension.", fileName.Data());
      return kFALSE;
   }

   if ( ! TakeLock(kDrawLock)) {
      Error(eh, "viewer locked - try later.");
      return kFALSE;
   }

   TUnlocker ulck(this);

   fLOD = TGLRnrCtx::kLODHigh;
   fRnrCtx->SetGrabImage(kTRUE);

   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoDraw(kFALSE)", (ULong_t)this));
   else
      DoDraw(kFALSE);

   fRnrCtx->SetGrabImage(kFALSE);

   glReadBuffer(GL_BACK);

   UChar_t* xx = new UChar_t[4 * fViewport.Width() * fViewport.Height()];
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, fViewport.Width(), fViewport.Height(),
                GL_BGRA, GL_UNSIGNED_BYTE, xx);

   std::auto_ptr<TImage> image(TImage::Create());
   image->FromGLBuffer(xx, fViewport.Width(), fViewport.Height());
   image->WriteImage(fileName);

   delete [] xx;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePictureUsingFBO(const TString &fileName, Int_t w, Int_t h,
                                      Float_t pixel_object_scale)
{
   // Save current image in various formats (gif, gif+, jpg, png).
   // 'gif+' will append image to an existng file (animated gif).
   // Frame-Buffer-Object is used for capturing of the image - OpenGL
   // 1.5 is required.
   // The viewer window does not have to be visible at all.
   // Returns false if something obvious goes wrong, true otherwise.
   //
   // pixel_object_scale is used to scale (as much as possible) the
   // objects whose representation size is pixel based (point-sizes,
   // line-widths, bitmap/pixmap font-sizes).
   // If set to 0 (default) no scaling is applied.

   static const TString eh("TGLViewer::SavePictureUsingFBO");

   if (! fileName.EndsWith(".gif") && ! fileName.Contains(".gif+") &&
       ! fileName.EndsWith(".jpg") && ! fileName.EndsWith(".png"))
   {
      Warning(eh, "file %s cannot be saved with this extension.", fileName.Data());
      return kFALSE;
   }

   if ( ! TakeLock(kDrawLock)) {
      Error(eh, "viewer locked - try later.");
      return kFALSE;
   }

   TUnlocker ulck(this);

   MakeCurrent();

   TGLFBO *fbo = new TGLFBO();
   try
   {
      fbo->Init(w, h, fGLWidget->GetPixelFormat()->GetSamples());
   }
   catch (std::runtime_error& exc)
   {
      Error(eh, "%s",exc.what());
      return kFALSE;
   }

   TGLRect old_vp(fViewport);
   SetViewport(0, 0, w, h);

   Float_t old_scale = 1;
   if (pixel_object_scale != 0)
   {
      old_scale = fRnrCtx->GetRenderScale();
      fRnrCtx->SetRenderScale(old_scale * pixel_object_scale);
   }

   fbo->Bind();

   fLOD = TGLRnrCtx::kLODHigh;
   fRnrCtx->SetGrabImage(kTRUE);

   if (!gVirtualX->IsCmdThread())
      gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoDraw(kFALSE)", (ULong_t)this));
   else
      DoDraw(kFALSE);

   fRnrCtx->SetGrabImage(kFALSE);

   fbo->Unbind();

   fbo->SetAsReadBuffer();

   UChar_t* xx = new UChar_t[4 * fViewport.Width() * fViewport.Height()];
   glPixelStorei(GL_PACK_ALIGNMENT, 1);
   glReadPixels(0, 0, fViewport.Width(), fViewport.Height(),
                GL_BGRA, GL_UNSIGNED_BYTE, xx);

   std::auto_ptr<TImage> image(TImage::Create());
   image->FromGLBuffer(xx, fViewport.Width(), fViewport.Height());
   image->WriteImage(fileName);

   delete [] xx;

   delete fbo;

   if (pixel_object_scale != 0)
   {
      fRnrCtx->SetRenderScale(old_scale);
   }

   SetViewport(old_vp);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePictureWidth(const TString &fileName, Int_t width,
                                   Bool_t pixel_object_scale)
{
   // Save picture with given width (height scaled proportinally).
   // If pixel_object_scale is true (default), the corresponding
   // scaling gets calculated from the current window size.

   Float_t scale  = Float_t(width) / fViewport.Width();
   Int_t   height = TMath::Nint(scale*fViewport.Height());

   return SavePictureUsingFBO(fileName, width, height, pixel_object_scale ? scale : 0);
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePictureHeight(const TString &fileName, Int_t height,
                                    Bool_t pixel_object_scale)
{
   // Save picture with given height (width scaled proportinally).
   // If pixel_object_scale is true (default), the corresponding
   // scaling gets calculated from the current window size.

   Float_t scale = Float_t(height) / fViewport.Height();
   Int_t   width = TMath::Nint(scale*fViewport.Width());

   return SavePictureUsingFBO(fileName, width, height, pixel_object_scale ? scale : 0);
}

//______________________________________________________________________________
Bool_t TGLViewer::SavePictureScale (const TString &fileName, Float_t scale,
                                    Bool_t pixel_object_scale)
{
   // Save picture with given scale to current window size.
   // If pixel_object_scale is true (default), the same scaling is
   // used.

   Int_t w = TMath::Nint(scale*fViewport.Width());
   Int_t h = TMath::Nint(scale*fViewport.Height());

   return SavePictureUsingFBO(fileName, w, h, pixel_object_scale ? scale : 0);
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
      const UChar_t rgba[4] = { 0, 255, 255, 255 };
      TGLUtil::DrawSphere(fCamera->GetCenterVec(), radius, rgba);
      disabled = kTRUE;
   }
   if (fAxesDepthTest && disabled)
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
   if (disabled)
      glEnable(GL_DEPTH_TEST);
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
      TGLUtil::DrawSphere(TGLVertex3(0.0, 0.0, 0.0), size, TGLUtil::fgWhite);
      const TGLVertex3 & center = fOverallBoundingBox.Center();
      TGLUtil::DrawSphere(center, size, TGLUtil::fgGreen);
      glEnable(GL_DEPTH_TEST);

      glEnable(GL_LIGHTING);
   }
}

//______________________________________________________________________________
void TGLViewer::PreDraw()
{
   // Perform GL work which must be done before each draw.

   InitGL();

   // For embedded gl clear color must be pad's background color.
   {
      Color_t ci = (fGLDevice != -1) ? gPad->GetFillColor() : fRnrCtx->ColorSet().Background().GetColorIndex();
      TColor *color = gROOT->GetColor(ci);
      Float_t rgb[3];
      if (color)
         color->GetRGB(rgb[0], rgb[1], rgb[2]);
      else
         rgb[0] = rgb[1] = rgb[2] = 0.0f;

      glClearColor(rgb[0], rgb[1], rgb[2], 0.0f);
   }

   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

   TGLUtil::CheckError("TGLViewer::PreDraw");
}

//______________________________________________________________________________
void TGLViewer::PostDraw()
{
   // Perform GL work which must be done after each draw.

   glFlush();
   TGLUtil::CheckError("TGLViewer::PostDraw");
}

//______________________________________________________________________________
void TGLViewer::FadeView(Float_t alpha)
{
   // Draw a rectangle (background color and given alpha) across the
   // whole viewport.

   static const Float_t z = -1.0f;

   glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
   glMatrixMode(GL_MODELVIEW);  glPushMatrix(); glLoadIdentity();

   {
      TGLCapabilitySwitch blend(GL_BLEND,    kTRUE);
      TGLCapabilitySwitch light(GL_LIGHTING, kFALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      TGLUtil::ColorAlpha(fRnrCtx->ColorSet().Background(), alpha);
      glBegin(GL_QUADS);
      glVertex3f(-1, -1, z);  glVertex3f( 1, -1, z);
      glVertex3f( 1,  1, z);  glVertex3f(-1,  1, z);
      glEnd();
   }

   glMatrixMode(GL_PROJECTION); glPopMatrix();
   glMatrixMode(GL_MODELVIEW);  glPopMatrix();
}

//______________________________________________________________________________
void TGLViewer::MakeCurrent() const
{
   // Make GL context current
   if (fGLDevice == -1)
      fGLWidget->MakeCurrent();
   else
      gGLManager->MakeCurrent(fGLDevice);
}

//______________________________________________________________________________
void TGLViewer::SwapBuffers() const
{
   // Swap GL buffers
   if ( ! IsDrawOrSelectLock()) {
      Error("TGLViewer::SwapBuffers", "viewer is %s", LockName(CurrentLock()));
   }
   if (fGLDevice == -1)
      fGLWidget->SwapBuffers();
   else {
      gGLManager->ReadGLBuffer(fGLDevice);
      gGLManager->Flush(fGLDevice);
      gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
   }
}

//______________________________________________________________________________
Bool_t TGLViewer::RequestSelect(Int_t x, Int_t y)
{
   // Post request for selection render pass viewer, picking objects
   // around the window point (x,y).

   // Take select lock on scene immediately we enter here - it is released
   // in the other (drawing) thread - see TGLViewer::DoSelect()

   if ( ! TakeLock(kSelectLock)) {
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoSelect(%d, %d)", (ULong_t)this, x, y)));
   else
      return DoSelect(x, y);
}

//______________________________________________________________________________
Bool_t TGLViewer::DoSelect(Int_t x, Int_t y)
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

   TUnlocker ulck(this);

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

   ReleaseLock(kSelectLock);
   return ! TGLSelectRecord::AreSameSelectionWise(fSelRec, fCurrentSelRec);
}

//______________________________________________________________________________
Bool_t TGLViewer::RequestSecondarySelect(Int_t x, Int_t y)
{
   // Request secondary select.

   if ( ! TakeLock(kSelectLock)) {
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoSecondarySelect(%d, %d)", (ULong_t)this, x, y)));
   else
      return DoSecondarySelect(x, y);
}

//______________________________________________________________________________
Bool_t TGLViewer::DoSecondarySelect(Int_t x, Int_t y)
{
   // Secondary selection.

   if (CurrentLock() != kSelectLock) {
      Error("TGLViewer::DoSecondarySelect", "expected kSelectLock, found %s", LockName(CurrentLock()));
      return kFALSE;
   }

   TUnlocker ulck(this);

   if (! fSelRec.GetSceneInfo() || ! fSelRec.GetPhysShape() ||
       ! fSelRec.GetPhysShape()->GetLogical()->SupportsSecondarySelect())
   {
      if (gDebug > 0)
         Info("TGLViewer::SecondarySelect", "Skipping secondary selection "
              "(sinfo=0x%lx, pshape=0x%lx).\n",
              (Long_t)fSelRec.GetSceneInfo(), (Long_t)fSelRec.GetPhysShape());
      fSecSelRec.Reset();
      return kFALSE;
   }

   MakeCurrent();

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

//______________________________________________________________________________
void TGLViewer::ApplySelection()
{
   // Process result from last selection (in fSelRec) and
   // extract a new current selection from it.
   // Here we only use physical shape.

   fCurrentSelRec = fSelRec;

   TGLPhysicalShape *selPhys = fSelRec.GetPhysShape();
   fSelectedPShapeRef->SetPShape(selPhys);

   // Inform external client selection has been modified.
   SelectionChanged();

   RequestDraw(TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
Bool_t TGLViewer::RequestOverlaySelect(Int_t x, Int_t y)
{
   // Post request for secondary selection rendering of selected object
   // around the window point (x,y).

   // Take select lock on viewer immediately - it is released
   // in the other (drawing) thread - see TGLViewer::DoSecondarySelect().

   if ( ! TakeLock(kSelectLock)) {
      return kFALSE;
   }

   if (!gVirtualX->IsCmdThread())
      return Bool_t(gROOT->ProcessLineFast(Form("((TGLViewer *)0x%lx)->DoOverlaySelect(%d, %d)", (ULong_t)this, x, y)));
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

   TUnlocker ulck(this);

   MakeCurrent();

   fRnrCtx->BeginSelection(x, y, 3);
   glRenderMode(GL_SELECT);

   PreRenderOverlaySelection();
   RenderOverlay(TGLOverlayElement::kActive, kTRUE);
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
         ++idx;
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

//______________________________________________________________________________
void TGLFaderHelper::MakeFadeStep()
{
   // Make one fading step and request redraw.

   Float_t fade = fViewer->GetFader();

   if (fade == fFadeTarget) {
      delete this; return;
   }
   if (TMath::Abs(fFadeTarget - fade) < 1e-3) {
      fViewer->SetFader(fFadeTarget);
      fViewer->RequestDraw(TGLRnrCtx::kLODHigh);
      delete this;
      return;
   }

   Float_t dt = fTime/fNSteps;
   Float_t df = (fFadeTarget - fade)/fNSteps;
   fViewer->SetFader(fade + df);
   fViewer->RequestDraw(TGLRnrCtx::kLODHigh);
   fTime -= dt; --fNSteps;
   TTimer::SingleShot(TMath::CeilNint(1000*dt),
                      "TGLFaderHelper", this, "MakeFadeStep()");
}

//______________________________________________________________________________
void TGLViewer::AutoFade(Float_t fade, Float_t time, Int_t steps)
{
   // Animate fading from curernt value to fade over given time (sec)
   // and number of steps.

   TGLFaderHelper* fh = new TGLFaderHelper(this, fade, time, steps);
   fh->MakeFadeStep();
}

//______________________________________________________________________________
void TGLViewer::UseDarkColorSet()
{
   // Use the dark color-set.

   fRnrCtx->ChangeBaseColorSet(&fDarkColorSet);
   RefreshPadEditor(this);
}

//______________________________________________________________________________
void TGLViewer::UseLightColorSet()
{
   // Use the light color-set.

   fRnrCtx->ChangeBaseColorSet(&fLightColorSet);
   RefreshPadEditor(this);
}

//______________________________________________________________________________
void TGLViewer::SwitchColorSet()
{
   // Swtich between dark and light colorsets.

   if (IsUsingDefaultColorSet())
   {
      Info("SwitchColorSet()", "Global color-set is in use, switch not supported.");
      return;
   }

   if (fRnrCtx->GetBaseColorSet() == &fLightColorSet)
      UseDarkColorSet();
   else
      UseLightColorSet();
}

//______________________________________________________________________________
void TGLViewer::UseDefaultColorSet(Bool_t x)
{
   // Set usage of the default color set.

   if (x)
      fRnrCtx->ChangeBaseColorSet(&fgDefaultColorSet);
   else
      fRnrCtx->ChangeBaseColorSet(&fDarkColorSet);
   RefreshPadEditor(this);
}

//______________________________________________________________________________
Bool_t TGLViewer::IsUsingDefaultColorSet() const
{
   // Check if the viewer is using the default color set.
   // If yes, some operations might be disabled.

   return fRnrCtx->GetBaseColorSet() == &fgDefaultColorSet;
}

//______________________________________________________________________________
void TGLViewer::SetClearColor(Color_t col)
{
   // Set background method.
   // Deprecated method - set background color in the color-set.

   fRnrCtx->GetBaseColorSet()->Background().SetColor(col);
}

//______________________________________________________________________________
TGLColorSet& TGLViewer::GetDefaultColorSet()
{
   // Returns reference to the default color-set.
   // Static function.

   return fgDefaultColorSet;
}

//______________________________________________________________________________
void TGLViewer::UseDefaultColorSetForNewViewers(Bool_t x)
{
   // Sets static flag that determines if new viewers should use the
   // default color-set.
   // This is false at startup.

   fgUseDefaultColorSetForNewViewers = x;
}

//______________________________________________________________________________
Bool_t TGLViewer::IsUsingDefaultColorSetForNewViewers()
{
   // Returns the value of the static flag that determines if new
   // viewers should use the default color-set.
   // This is false at startup.

   return fgUseDefaultColorSetForNewViewers;
}

//______________________________________________________________________________
Bool_t TGLViewer::IsColorSetDark() const
{
   // Returns true if curremt color set is dark.

   return fRnrCtx->GetBaseColorSet() == &fDarkColorSet;
}

/**************************************************************************/
// Viewport
/**************************************************************************/

//______________________________________________________________________________
void TGLViewer::SetViewport(Int_t x, Int_t y, Int_t width, Int_t height)
{
   // Set viewer viewport (window area) with bottom/left at (x,y), with
   // dimensions 'width'/'height'

   // Only process if changed
   if (fViewport.X() == x && fViewport.Y() == y &&
       fViewport.Width() == width && fViewport.Height() == height) {
      return;
   }

   fViewport.Set(x, y, width, height);
   fCurrentCamera->SetViewport(fViewport);

   if (gDebug > 2) {
      Info("TGLViewer::SetViewport", "updated - corner %d,%d dimensions %d,%d", x, y, width, height);
   }
}

void TGLViewer::SetViewport(const TGLRect& vp)
{
   // Set viewr viewport from TGLRect.

   SetViewport(vp.X(), vp.Y(), vp.Width(), vp.Height());
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
      case kCameraOrthoXnOY:
         return fOrthoXnOYCamera;
      case kCameraOrthoXnOZ:
         return fOrthoXnOZCamera;
      case kCameraOrthoZnOY:
         return fOrthoZnOYCamera;
      default:
         Error("TGLViewer::SetCurrentCamera", "invalid camera type");
         return *fCurrentCamera;
   }
}

//______________________________________________________________________________
void TGLViewer::SetCurrentCamera(ECameraType cameraType)
{
   // Set current active camera - 'cameraType' one of:
   //   kCameraPerspX,    kCameraPerspY,    kCameraPerspZ,
   //   kCameraOrthoXOY,  kCameraOrthoXOZ,  kCameraOrthoZOY,
   //   kCameraOrthoXnOY, kCameraOrthoXnOZ, kCameraOrthoZnOY

   if (IsLocked()) {
      Error("TGLViewer::SetCurrentCamera", "expected kUnlocked, found %s", LockName(CurrentLock()));
      return;
   }

   // TODO: Move these into a vector!
   TGLCamera *prev = fCurrentCamera;
   switch (cameraType)
   {
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
      case kCameraOrthoXnOY: {
         fCurrentCamera = &fOrthoXnOYCamera;
         break;
      }
      case kCameraOrthoXnOZ: {
         fCurrentCamera = &fOrthoXnOZCamera;
         break;
      }
      case kCameraOrthoZnOY: {
         fCurrentCamera = &fOrthoZnOYCamera;
         break;
      }
      default: {
         Error("TGLViewer::SetCurrentCamera", "invalid camera type");
         break;
      }
   }

   if (fCurrentCamera != prev)
   {
      // Ensure any viewport has been propigated to the current camera
      fCurrentCamera->SetViewport(fViewport);
      RefreshPadEditor(this);

      if (fAutoRotator)
      {
         if (fAutoRotator->IsRunning())
         {
            fAutoRotator->Stop();
         }
         else
         {
            if (fAutoRotator->GetCamera() == fCurrentCamera)
            {
               fAutoRotator->Start();
            }
         }
      }

      RequestDraw(TGLRnrCtx::kLODHigh);
   }
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

//______________________________________________________________________________
TGLAutoRotator* TGLViewer::GetAutoRotator()
{
   // Get the auto-rotator for this viewer.

   if (fAutoRotator == 0)
      fAutoRotator = new TGLAutoRotator(this);
   return fAutoRotator;
}

//______________________________________________________________________________
void TGLViewer::SetAutoRotator(TGLAutoRotator* ar)
{
   // Set the auto-rotator for this viewer. The old rotator is deleted.

   delete fAutoRotator;
   fAutoRotator = ar;
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
   if (referencePos)
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
void TGLViewer::MouseOver(TGLPhysicalShape *shape)
{
   // Emit MouseOver signal.

   Emit("MouseOver(TGLPhysicalShape*)", (Long_t)shape);
}

//______________________________________________________________________________
void TGLViewer::MouseOver(TGLPhysicalShape *shape, UInt_t state)
{
   // Emit MouseOver signal.

   Long_t args[2];
   args[0] = (Long_t)shape;
   args[1] = state;
   Emit("MouseOver(TGLPhysicalShape*,UInt_t)", args);
}

//______________________________________________________________________________
void TGLViewer::MouseOver(TObject *obj, UInt_t state)
{
   // Emit MouseOver signal.

   Long_t args[2];
   args[0] = (Long_t)obj;
   args[1] = state;
   Emit("MouseOver(TObject*,UInt_t)", args);
}

//______________________________________________________________________________
void TGLViewer::ReMouseOver(TObject *obj, UInt_t state)
{
   // Emit MouseOver signal.

   Long_t args[2];
   args[0] = (Long_t)obj;
   args[1] = state;
   Emit("ReMouseOver(TObject*,UInt_t)", args);
}


//______________________________________________________________________________
void TGLViewer::UnMouseOver(TObject *obj, UInt_t state)
{
   // Emit UnMouseOver signal.

   Long_t args[2];
   args[0] = (Long_t)obj;
   args[1] = state;
   Emit("UnMouseOver(TObject*,UInt_t)", args);
}

//______________________________________________________________________________
void TGLViewer::Clicked(TObject *obj)
{
   // Emit Clicked signal.

   Emit("Clicked(TObject*)", (Long_t)obj);
}

//______________________________________________________________________________
void TGLViewer::Clicked(TObject *obj, UInt_t button, UInt_t state)
{
   // Emit Clicked signal with button id and modifier state.

   Long_t args[3];
   args[0] = (Long_t)obj;
   args[1] = button;
   args[2] = state;
   Emit("Clicked(TObject*,UInt_t,UInt_t)", args);
}


//______________________________________________________________________________
void TGLViewer::ReClicked(TObject *obj, UInt_t button, UInt_t state)
{
   // Emit ReClicked signal with button id and modifier state.

   Long_t args[3];
   args[0] = (Long_t)obj;
   args[1] = button;
   args[2] = state;
   Emit("ReClicked(TObject*,UInt_t,UInt_t)", args);
}

//______________________________________________________________________________
void TGLViewer::UnClicked(TObject *obj, UInt_t button, UInt_t state)
{
   // Emit UnClicked signal with button id and modifier state.

   Long_t args[3];
   args[0] = (Long_t)obj;
   args[1] = button;
   args[2] = state;
   Emit("UnClicked(TObject*,UInt_t,UInt_t)", args);
}

//______________________________________________________________________________
void TGLViewer::MouseIdle(TGLPhysicalShape *shape, UInt_t posx, UInt_t posy)
{
   // Emit MouseIdle signal.

   Long_t args[3];
   static UInt_t oldx = 0, oldy = 0;

   if (oldx != posx || oldy != posy) {
      args[0] = (Long_t)shape;
      args[1] = posx;
      args[2] = posy;
      Emit("MouseIdle(TGLPhysicalShape*,UInt_t,UInt_t)", args);
      oldx = posx;
      oldy = posy;
   }
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

   if (fEventHandler)
      return fEventHandler->ExecuteEvent(event, px, py);
}

//______________________________________________________________________________
void TGLViewer::PrintObjects()
{
   // Pass viewer for print capture by TGLOutput.

   TGLOutput::Capture(*this);
}

//______________________________________________________________________________
void TGLViewer::SelectionChanged()
{
   // Update GUI components for embedded viewer selection change.

   if (!fGedEditor)
      return;

   TGLPhysicalShape *selected = const_cast<TGLPhysicalShape*>(GetSelected());

   if (selected) {
      fPShapeWrap->fPShape = selected;
      fGedEditor->SetModel(fPad, fPShapeWrap, kButton1Down);
   } else {
      fPShapeWrap->fPShape = 0;
      fGedEditor->SetModel(fPad, this, kButton1Down);
   }
}

//______________________________________________________________________________
void TGLViewer::OverlayDragFinished()
{
   // An overlay operation can result in change to an object.
   // Refresh geditor.

   if (fGedEditor)
   {
      fGedEditor->SetModel(fPad, fGedEditor->GetModel(), kButton1Down);
   }
}

//______________________________________________________________________________
void TGLViewer::RefreshPadEditor(TObject* obj)
{
   // Update GED editor if it is set.

   if (fGedEditor && (obj == 0 || fGedEditor->GetModel() == obj))
   {
      fGedEditor->SetModel(fPad, fGedEditor->GetModel(), kButton1Down);
   }
}

//______________________________________________________________________________
void TGLViewer::SetEventHandler(TGEventHandler *handler)
{
   // Set the event-handler. The event-handler is owned by the viewer.
   // If GLWidget is set, the handler is propagated to it.
   //
   // If called with handler=0, the current handler will be deleted
   // (also from TGLWidget).

   if (fEventHandler)
      delete fEventHandler;

   fEventHandler = handler;
   if (fGLWidget)
      fGLWidget->SetEventHandler(fEventHandler);
}

//______________________________________________________________________________
void  TGLViewer::RemoveOverlayElement(TGLOverlayElement* el)
{
   // Remove overlay element.

   if (el == fCurrentOvlElm)
   {
      fCurrentOvlElm = 0;
   }
   TGLViewerBase::RemoveOverlayElement(el);
}

//______________________________________________________________________________
void TGLViewer::ClearCurrentOvlElm()
{
   // Reset current overlay-element to zero, eventually notifying the
   // old one that the mouse has left.
   // Usually called when mouse leaves the window.

   if (fCurrentOvlElm)
   {
      fCurrentOvlElm->MouseLeave();
      fCurrentOvlElm = 0;
      RequestDraw();
   }
}
