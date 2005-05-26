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
#include "TGLDisplayListCache.h" // TODO: Testing - remove

#include "Riostream.h" // TODO: Testing - remove

// TODO: Find a better place/way to do this
class TGLRedrawTimer : public TTimer
{
   private:
      TGLViewer & fViewer;
   public:
      TGLRedrawTimer(TGLViewer & viewer) : fViewer(viewer) {};
      ~TGLRedrawTimer() {};
      Bool_t Notify() { TurnOff(); fViewer.Invalidate(kHigh); return kTRUE; }
};

ClassImp(TGLViewer)

//______________________________________________________________________________
TGLViewer::TGLViewer() :
   fNextSceneLOD(kHigh),
   fCurrentCamera(&fPerspectiveCamera), 
   fPerspectiveCamera(),
   fOrthoXOYCamera(TGLOrthoCamera::kXOY),
   fOrthoYOZCamera(TGLOrthoCamera::kYOZ),
   fOrthoXOZCamera(TGLOrthoCamera::kXOZ),
   fInitGL(kFALSE)
{
   fRedrawTimer = new TGLRedrawTimer(*this);
}

//______________________________________________________________________________
TGLViewer::~TGLViewer()
{
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
void TGLViewer::Draw()
{
   //TGLStopwatch timer;
   //timer.Start();

   PreDraw();

   // Something to draw?
   if (!fScene.BoundingBox().IsEmpty()) {
      // Apply current camera projection
      fCurrentCamera->Apply(fScene.BoundingBox());

      // TODO: Drop objects below a certain (projected) size?
      if (fNextSceneLOD == kHigh) {
         fScene.Draw(*fCurrentCamera, fNextSceneLOD);
      } else {
         // Force redraw to terminate after 300 msec - needs more playing with - based
         // on multiple scene qualities?
         fScene.Draw(*fCurrentCamera, fNextSceneLOD, 300.0);
      }
   }
   
   PostDraw();
}

//______________________________________________________________________________
void TGLViewer::PostDraw()
{   
   // GL work which must be done after each draw of scene
   SwapBuffers();   

   // Flush everything in case picking starts
   glFlush();

   // Rebuild scene?
   if (fNextSceneLOD == kHigh && CurrentCamera().UpdateInterest()) {
      RebuildScene();
   } else if (fNextSceneLOD != kHigh) {
      // Final pass render required
      // TODO: Should really be another factor on top of scene draw qaulity   
      // 100 msec single shot callback to redraw at best quality if no
      // other user action during this time
      fRedrawTimer->Start(100,kTRUE);
   }
  
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
   TGLRect glRect(rect);
   WindowToGL(glRect);
   fCurrentCamera->Apply(fScene.BoundingBox(), &glRect);
 
   MakeCurrent();
   Bool_t changed = fScene.Select(*fCurrentCamera);
   if (changed) {
      Invalidate(kHigh);
   }
   return changed;
}

//______________________________________________________________________________
void TGLViewer::SetViewport(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   fViewport.Set(x, y, width, height);
   fCurrentCamera->SetViewport(fViewport);
   Invalidate();
}

//______________________________________________________________________________
void TGLViewer::SetCurrentCamera(ECamera camera)
{
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
   fPerspectiveCamera.Setup(box);
   fOrthoXOYCamera.Setup(box);
   fOrthoYOZCamera.Setup(box);
   fOrthoXOZCamera.Setup(box);
}

