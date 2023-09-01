// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLAutoRotator.h"

#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#include "TGLScene.h"

#include "TMath.h"
#include "TTimer.h"
#include "TStopwatch.h"

/** \class TGLAutoRotator
\ingroup opengl
Automatically rotates GL camera.

W's are angular velocities.
  - ATheta -- Theta amplitude in units of Pi/2.
  - ADolly -- In/out amplitude in units of initial distance.

Can also save images automatically.

fGUIOutMode is used internally between TGLAutoRotator and TGLViewerEditor,
allowed values are:
  1. animated gif
  2. a series of png images
*/

ClassImp(TGLAutoRotator);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLAutoRotator::TGLAutoRotator(TGLViewer* v) :
   fViewer(v), fCamera(0),
   fTimer(new TTimer), fWatch(new TStopwatch),
   fRotateScene(kFALSE),
   fDeltaPhi(0.005),
   fDt    (0.01),
   fWPhi  (0.40),
   fWTheta(0.15), fATheta(0.5),
   fWDolly(0.30), fADolly(0.4),
   fTimerRunning(kFALSE),
   fImageCount(0), fImageAutoSave(kFALSE),
   fImageGUIBaseName("animation"), fImageGUIOutMode(1)
{
   fTimer->Connect("Timeout()", "TGLAutoRotator", this, "Timeout()");
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLAutoRotator::~TGLAutoRotator()
{
   delete fWatch;
   delete fTimer;
}

////////////////////////////////////////////////////////////////////////////////
/// Set time between two redraws in seconds.
/// Range: 0.001 -> 1.

void TGLAutoRotator::SetDt(Double_t dt)
{
   fDt = TMath::Range(0.01, 1.0, dt);
   if (fTimerRunning)
   {
      fTimer->SetTime(TMath::Nint(1000*fDt));
      fTimer->Reset();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set relative amplitude of theta oscillation.
/// Value range: 0.01 -> 1.

void TGLAutoRotator::SetATheta(Double_t a)
{
   a = TMath::Range(0.01, 1.0, a);
   if (fTimerRunning)
   {
      fThetaA0 = fThetaA0 * a / fATheta;
   }
   fATheta = a;
}

////////////////////////////////////////////////////////////////////////////////
/// Set relative amplitude of forward/backward oscillation.
/// Value range: 0.01 -> 1.

void TGLAutoRotator::SetADolly(Double_t a)
{
  a = TMath::Range(0.01, 1.0, a);
  if (fTimerRunning)
  {
     fDollyA0 = fDollyA0 * a / fADolly;
  }
  fADolly = a;
}

////////////////////////////////////////////////////////////////////////////////
/// Start the auto-rotator.

void TGLAutoRotator::Start()
{
   if (fTimerRunning)
   {
      Stop();
   }

   fCamera = & fViewer->CurrentCamera();

   fThetaA0 = fATheta * TMath::PiOver2();
   fDollyA0 = fADolly * fCamera->GetCamTrans().GetBaseVec(4).Mag();

   fTimerRunning = kTRUE;
   fTimer->SetTime(TMath::Nint(1000*fDt));
   fTimer->Reset();
   fTimer->TurnOn();
   fWatch->Start();
}

////////////////////////////////////////////////////////////////////////////////
/// Stop the auto-rotator.

void TGLAutoRotator::Stop()
{
   if (fTimerRunning)
   {
      fWatch->Stop();
      fTimer->TurnOff();
      fTimerRunning = kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Called on every timer timeout. Moves / rotates the camera and optionally
/// produces a screenshot.

void TGLAutoRotator::Timeout()
{
   if (!fTimerRunning || gTQSender != fTimer)
   {
      Error("Timeout", "Not running or not called via timer.");
      return;
   }

   using namespace TMath;

   fWatch->Stop();
   Double_t time = fWatch->RealTime();
   fWatch->Continue();

   if (fRotateScene) {
      RotateScene();
   } else {
      Double_t delta_p = fWPhi*fDt;
      Double_t delta_t = fThetaA0*fWTheta*Cos(fWTheta*time)*fDt;
      Double_t delta_d = fDollyA0*fWDolly*Cos(fWDolly*time)*fDt;
      Double_t th      = fCamera->GetTheta();

      if (th + delta_t > 3.0 || th + delta_t < 0.1416)
         delta_t = 0;

      fCamera->RotateRad(delta_t, delta_p);
      fCamera->RefCamTrans().MoveLF(1, -delta_d);
   }

   fViewer->RequestDraw(TGLRnrCtx::kLODHigh);

   if (fImageAutoSave)
   {
      TString filename;
      if (fImageName.Contains("%"))
      {
         filename.Form(fImageName, fImageCount);
      }
      else
      {
         filename = fImageName;
      }
      fViewer->SavePicture(filename);
      ++fImageCount;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Start saving into animated gif. The provided name will be used as it is,
/// so make sure to end it with '.gif+'.
/// Use convert tool from ImageMagic if you want to set a different delay or
/// enable looping.

void TGLAutoRotator::StartImageAutoSaveAnimatedGif(const TString& filename)
{
   if ( ! filename.Contains(".gif+"))
   {
      Error("StartImageAutoSaveAnimatedGif", "Name should end with '.gif+'. Not starting.");
      return;
   }

   fImageName     = filename;
   fImageCount    = 0;
   fImageAutoSave = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Start saving into a set of images. The provided name will be used as a
/// format to insert additional image sequence number so it should include
/// an '%' character. A good name would be something like:
///   "image-%04d.png"
/// On GNU/Linux use mencoder and/or ffmpeg to bundle images into a movie.

void TGLAutoRotator::StartImageAutoSave(const TString& filename)
{
   if ( ! filename.Contains("%"))
   {
      Error("StartImageAutoSave", "Name should include a '%%' character, like 'image-%%05d.png'. Not starting.");
      return;
   }

   fImageName     = filename;
   fImageCount    = 0;
   fImageAutoSave = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Stops automatic saving of images.

void TGLAutoRotator::StopImageAutoSave()
{
   fImageAutoSave = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set output mode for GUI operation:
///   1 - animated gif;
///   2 - a series of pngs

void TGLAutoRotator::SetImageGUIOutMode(Int_t m)
{
   if (m < 1 || m > 2)
   {
      Warning("SetImageGUIOutMode", "Invalid value, ignoring");
      return;
   }
   fImageGUIOutMode = m;
}

////////////////////////////////////////////////////////////////////////////////
/// Start auto-saving images as set-up via GUI.

void TGLAutoRotator::StartImageAutoSaveWithGUISettings()
{
   if (fImageGUIOutMode == 1)
   {
      TString name = fImageGUIBaseName + ".gif+";
      StartImageAutoSaveAnimatedGif(name);
   }
   else if (fImageGUIOutMode == 2)
   {
      TString name = fImageGUIBaseName + "-%05d.png";
      StartImageAutoSave(name);
   }
   else
   {
      Warning("StartImageAutoSaveWithGUISettings", "Unsupported mode '%d'.", fImageGUIOutMode);
   }
}

////////////////////////////////////////////////////////////////////////////////
///"Scene rotation": either find a special object,
///which will be an axis of rotation (it's Z actually)
///or use a "global" Z axis.

void TGLAutoRotator::RotateScene()
{
   TGLViewer::SceneInfoList_t & scenes = fViewer->fScenes;
   TGLViewer::SceneInfoList_i sceneIter = scenes.begin();

   for (; sceneIter != scenes.end(); ++sceneIter) {
     TGLScene::TSceneInfo *sceneInfo = dynamic_cast<TGLScene::TSceneInfo *>(*sceneIter);
      if (sceneInfo) {
         TGLPhysicalShape *axisShape = 0;
         TGLScene::ShapeVec_i shapeIter = sceneInfo->fShapesOfInterest.begin();
         for (; shapeIter != sceneInfo->fShapesOfInterest.end(); ++shapeIter) {
            TGLPhysicalShape * const testShape = const_cast<TGLPhysicalShape *>(*shapeIter);
            if (testShape && testShape->GetLogical()->ID()->TestBit(13)) {
               axisShape = testShape;
               break;
            }
         }

         TGLVector3 axis;
         TGLVertex3 center;

         if (!axisShape) {
            const TGLBoundingBox &bbox = sceneInfo->GetTransformedBBox();
            axis = bbox.Axis(2);
            center = bbox.Center();
         } else {
            axis = axisShape->BoundingBox().Axis(2);
            center = axisShape->BoundingBox().Center();
         }

         shapeIter = sceneInfo->fShapesOfInterest.begin();
         for (; shapeIter != sceneInfo->fShapesOfInterest.end(); ++shapeIter) {
            if (TGLPhysicalShape * const shape = const_cast<TGLPhysicalShape *>(*shapeIter))
               shape->Rotate(center, axis, fDeltaPhi);
         }
      }
   }
}
