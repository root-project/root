// @(#)root/gl:$Name:  $:$Id: TGLViewer.cxx,v 1.6 2005/06/15 15:40:30 brun Exp $
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

ClassImp(TGLViewer)

//______________________________________________________________________________
TGLViewer::TGLViewer() :
   fPerspectiveCamera(),
   fOrthoXOYCamera(TGLOrthoCamera::kXOY),
   fOrthoYOZCamera(TGLOrthoCamera::kYOZ),
   fOrthoXOZCamera(TGLOrthoCamera::kXOZ),
   fCurrentCamera(&fPerspectiveCamera),
   fRedrawTimer(0),
   fNextSceneLOD(kHigh),
   fClipPlane(1.0, 0.0, 0.0, 0.0),
   fUseClipPlane(kFALSE),
   fDrawAxes(kFALSE),
   fInitGL(kFALSE)
{
   fRedrawTimer = new TGLRedrawTimer(*this);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
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
         return;
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

   // Scene rebuild required?
   if (!RebuildScene()) {
      // Final draw pass required?
      if (fNextSceneLOD != kHigh) {
         fRedrawTimer->RequestDraw(100, kHigh);
      }
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
