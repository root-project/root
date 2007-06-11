//script showing how to use the GL viewer API to animate a picture
//Author: Richard maunder

#include "TGLViewer.h"
#include "TTimer.h"
#include "TRandom.h"
#include "TVirtualPad.h"

TGLViewer::ECameraType camera;
TTimer timer(50);
TRandom randGen(0);

void AnimatePerspectiveCamera()
{
   static Double_t fov = 50.0;
   static Double_t dolly = 1500.0;
   static Double_t center[3] = {-164.0, -164.0, -180.0};
   static Double_t hRotate = 0.0;
   static Double_t vRotate = 0.0;

   static Double_t fovStep = randGen.Rndm()*5.0 - 2.5;
   static Double_t dollyStep = randGen.Rndm()*10.0 - 5.0;
   static Double_t centerStep[3] = {randGen.Rndm()*20.0 - 10.0, 
                                    randGen.Rndm()*20.0 - 10.0, 
                                    randGen.Rndm()*20.0 - 10.0};
   static Double_t hRotateStep = randGen.Rndm()*10.0 - 5.0;
   static Double_t vRotateStep = randGen.Rndm()*10.0 - 5.0;

   fov += fovStep;
   dolly += dollyStep;
   center[0] += centerStep[0];
   center[1] += centerStep[1];
   center[2] += centerStep[2];
   hRotate += hRotateStep;
   vRotate += vRotateStep;

   if (vRotate >= 90.0 || vRotate <= -90.0) 
   {
   	vRotateStep = -vRotateStep;
   }
   if (dolly >= 2000.0 || dolly <= 1000.0) {
   	dollyStep = -dollyStep;
   }
   if (fov > 170.0 || fov < 1.0) {
      fovStep = - fovStep;
   }
   TGLViewer * v = (TGLViewer *)gPad->GetViewer3D();

   /*
   void  SetPerspectiveCamera(ECameraType camera, Double_t fov, Double_t dolly, 
                              Double_t center[3], Double_t hRotate, Double_t vRotate);*/
   v->SetPerspectiveCamera(camera, fov, dolly, center, hRotate, vRotate);
}

void AnimateOrthographicCamera()
{
   static Double_t left  = -100.0;
   static Double_t right = 100.0;
   static Double_t top   = 100.0;
   static Double_t bottom = -100.0;

   static Double_t leftStep  = randGen.Rndm()*40.0 - 10.0;
   static Double_t rightStep = randGen.Rndm()*20.0 - 10.0;
   static Double_t topStep   = randGen.Rndm()*40.0 - 10.0;
   static Double_t bottomStep = randGen.Rndm()*20.0 -10.0;

   left += leftStep;
   right += rightStep;
   top += topStep;
   bottom += bottomStep;

   if (left >= 0.0 || left <= -500.0) {
   	leftStep = -leftStep;
   }
   if (right >= 500.0 || right <= 0.0) {
   	rightStep = -rightStep;
   }
   if (top >= 500.0 || top <= 0.0) {
   	topStep = -topStep;
   }
   if (bottom >= 0.0 || bottom <= -500.0) {
   	bottomStep = -bottomStep;
   }

   TGLViewer * v = (TGLViewer *)gPad->GetViewer3D();

   /*
   void  SetOrthoCamera(ECameraType camera, Double_t left, Double_t right, Double_t top, Double_t bottom); */
   v->SetOrthoCamera(camera, left, right, top, bottom);
}

void glViewerExercise()
{
   gROOT->ProcessLine(".x nucleus.C");
   TGLViewer * v = (TGLViewer *)gPad->GetViewer3D();
   
   // Random draw style 
   Int_t style = randGen.Integer(3);
   switch (style) {
      case 0: v->SetStyle(TGLRnrCtx::kFill); break;
      case 1: v->SetStyle(TGLRnrCtx::kOutline); break;
      case 2: v->SetStyle(TGLRnrCtx::kWireFrame); break;
   }   

   // Clipping setup - something like this:
   /*
   Double_t planeEq[4] = { 0.5, 1.0, -1.0, 2.0 };
   v->SetClipState(TGLViewer::kClipPlane, planeEq);
   v->SetCurrentClip(TGLViewer::kClipPlane, kTRUE);
   */

   // Guides - something like this:
   /*
   Double_t refPos[3] = { 50.0, 60.0, 100.0 };
   v->SetGuideState(TGLViewer::kAxesEdge, kTRUE, refPos); 
   */

   // Lights - turn some off randomly
   TGLLightSet* ls = v->GetLightSet();
   if (randGen.Integer(2) == 0) {
      ls->SetLight(TGLLightSet::kLightLeft, kFALSE);
   }
   if (randGen.Integer(2) == 0) {
      ls->SetLight(TGLLightSet::kLightRight, kFALSE);
   }
   if (randGen.Integer(2) == 0) {
      ls->SetLight(TGLLightSet::kLightTop, kFALSE);
   }
   if (randGen.Integer(2) == 0) {
      ls->SetLight(TGLLightSet::kLightBottom, kFALSE);
   }

   // Random camera type
   /*
   enum ECameraType { kCameraPerspXOZ, kCameraPerspYOZ, kCameraPerspXOY,
                      kCameraOrthoXOY, kCameraOrthoXOZ, kCameraOrthoZOY };*/
   Int_t cam = randGen.Integer(6);
   camera = (TGLViewer::ECameraType)cam;
   v->SetCurrentCamera(camera);

   // Now animate the camera
   if (camera <3) {
      timer.SetCommand("AnimatePerspectiveCamera()");
   } else {
      timer.SetCommand("AnimateOrthographicCamera()");
   }
   timer.TurnOn();
}


