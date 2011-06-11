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

#include "TGLViewer.h"
#include "TGLCamera.h"

#include "TMath.h"
#include "TTimer.h"
#include "TStopwatch.h"

//______________________________________________________________________________
//
// Automatically rotates GL camera.
//
// W's are angular velocities.
// ATheta -- Theta amplitude in units of Pi/2.
// ADolly -- In/out amplitude in units of initial distance.
//
// Can also save images automatically.

// fGUIOutMode is used internally between TGLAutoRotator and TGLViewerEditor,
// allowed values are:
//   1 - animated gif
//   2 - a series of png images

ClassImp(TGLAutoRotator);

//______________________________________________________________________________
TGLAutoRotator::TGLAutoRotator(TGLViewer* v) :
   fViewer(v), fCamera(0),
   fTimer(new TTimer), fWatch(new TStopwatch),
   fDt    (0.01),
   fWPhi  (0.40),
   fWTheta(0.15), fATheta(0.5),
   fWDolly(0.30), fADolly(0.4),
   fTimerRunning(kFALSE),
   fImageCount(0), fImageAutoSave(kFALSE),
   fImageGUIBaseName("animation"), fImageGUIOutMode(1)
{
   // Constructor.

   fTimer->Connect("Timeout()", "TGLAutoRotator", this, "Timeout()");
}

//______________________________________________________________________________
TGLAutoRotator::~TGLAutoRotator()
{
   // Destructor.

   delete fWatch;
   delete fTimer;
}

//==============================================================================

//______________________________________________________________________________
void TGLAutoRotator::SetDt(Double_t dt)
{
   // Set time between two redraws in seconds.
   // Range: 0.001 -> 1.

   fDt = TMath::Range(0.01, 1.0, dt);
   if (fTimerRunning)
   {
      fTimer->SetTime(TMath::Nint(1000*fDt));
      fTimer->Reset();
   }
}

//______________________________________________________________________________
void TGLAutoRotator::SetATheta(Double_t a)
{
   // Set relative amplitude of theta oscilation.
   // Value range: 0.01 -> 1.

   a = TMath::Range(0.01, 1.0, a);
   if (fTimerRunning)
   {
      fThetaA0 = fThetaA0 * a / fATheta;
   }
   fATheta = a;
}

//______________________________________________________________________________
void TGLAutoRotator::SetADolly(Double_t a)
{
   // Set relative amplitude of forward/backward oscilation.
   // Value range: 0.01 -> 1.

  a = TMath::Range(0.01, 1.0, a);
  if (fTimerRunning)
  {
     fDollyA0 = fDollyA0 * a / fADolly;
  }
  fADolly = a;
}

//==============================================================================

//______________________________________________________________________________
void TGLAutoRotator::Start()
{
   // Start the auto-rotator.

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

//______________________________________________________________________________
void TGLAutoRotator::Stop()
{
   // Stop the auto-rotator.

   if (fTimerRunning)
   {
      fWatch->Stop();
      fTimer->TurnOff();
      fTimerRunning = kFALSE;
   }
}

//______________________________________________________________________________
void TGLAutoRotator::Timeout()
{
   // Called on every timer timeout. Moves / rotates the camera and optionally
   // produces a screenshot.

   if (!fTimerRunning || gTQSender != fTimer)
   {
      Error("Timeout", "Not running or not called via timer.");
      return;
   }

   using namespace TMath;

   fWatch->Stop();
   Double_t time = fWatch->RealTime();
   fWatch->Continue();

   Double_t delta_p = fWPhi*fDt;
   Double_t delta_t = fThetaA0*fWTheta*Cos(fWTheta*time)*fDt;
   Double_t delta_d = fDollyA0*fWDolly*Cos(fWDolly*time)*fDt;
   Double_t th      = fCamera->GetTheta();

   if (th + delta_t > 3.0 || th + delta_t < 0.1416)
      delta_t = 0;

   fCamera->RotateRad(delta_t, delta_p);
   fCamera->RefCamTrans().MoveLF(1, -delta_d);

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

//==============================================================================

//______________________________________________________________________________
void TGLAutoRotator::StartImageAutoSaveAnimatedGif(const TString& filename)
{
   // Start saving into animated gif. The provided name will be used as it is,
   // so make sure to end it with '.gif+'.
   // Use convert tool from ImageMagic if you want to set a different delay or
   // enable looping.

   if ( ! filename.Contains(".gif+"))
   {
      Error("StartImageAutoSaveAnimatedGif", "Name should end with '.gif+'. Not starting.");
      return;
   }

   fImageName     = filename;
   fImageCount    = 0;
   fImageAutoSave = kTRUE;
}

//______________________________________________________________________________
void TGLAutoRotator::StartImageAutoSave(const TString& filename)
{
   // Start saving into a set of images. The provided name will be used as a
   // format to insert additional image sequence number so it should include
   // an '%' character. A good name would be something like:
   //   "image-%04d.png"
   // On GNU/Linux use mencoder and/or ffmpeg to bundle images into a movie.

   if ( ! filename.Contains("%"))
   {
      Error("StartImageAutoSave", "Name should include a '%%' character, like 'image-%%05d.png'. Not starting.");
      return;
   }

   fImageName     = filename;
   fImageCount    = 0;
   fImageAutoSave = kTRUE;
}

//______________________________________________________________________________
void TGLAutoRotator::StopImageAutoSave()
{
   // Stops automatic saving of images.

   fImageAutoSave = kFALSE;
}

//______________________________________________________________________________
void TGLAutoRotator::SetImageGUIOutMode(Int_t m)
{
   // Set output mode for GUI operation:
   //   1 - animated gif;
   //   2 - a series of pngs

   if (m < 1 || m > 2)
   {
      Warning("SetImageGUIOutMode", "Invalid value, ignoring");
      return;
   }
   fImageGUIOutMode = m;
}

//______________________________________________________________________________
void TGLAutoRotator::StartImageAutoSaveWithGUISettings()
{
   // Start auto-saving images as set-up via GUI.

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
