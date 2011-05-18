// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLLightSet.h"

#include "TGLBoundingBox.h"
#include "TGLOrthoCamera.h"

#include "TGLIncludes.h"

//______________________________________________________________________
// TGLLightSet
//
// Encapsulates a set of lights for OpenGL.
//

ClassImp(TGLLightSet)

TGLLightSet::TGLLightSet() :
   TObject(),

   fLightState(kLightMask), // All on
   fUseSpecular(kTRUE),

   fFrontPower(0.4), fSidePower(0.7), fSpecularPower(0.8)
{
   // Constructor.
}

//______________________________________________________________________________
void TGLLightSet::ToggleLight(ELight light)
{
   // Toggle light on/off.

   if (light == kLightSpecular) {
      fUseSpecular = !fUseSpecular;
   } else if (light >= kLightMask) {
      Error("TGLLightSet::ToggleLight", "invalid light type");
      return;
   } else {
      fLightState ^= light;
   }
}

//______________________________________________________________________________
void TGLLightSet::SetLight(ELight light, Bool_t on)
{
   // Set a light on/off.

   if (light == kLightSpecular) {
      fUseSpecular = on;
   } else if (light >= kLightMask) {
      Error("TGLViewer::ToggleLight", "invalid light type");
      return;
   }

   if (on) {
      fLightState |=  light;
   } else {
      fLightState &= ~light;
   }
}

//______________________________________________________________________________
void TGLLightSet::StdSetupLights(const TGLBoundingBox& bbox,
                                 const TGLCamera     & camera, Bool_t debug)
{
   // Setup lights for current given bounding box and camera.
   // This is called by standard GL viewer.
   // Expects matrix-mode to be model-view.

   glPushMatrix();

   if (!bbox.IsEmpty())
   {
      // Calculate a sphere radius to arrange lights round
      Double_t lightRadius = bbox.Extents().Mag() * 2.9;
      Double_t sideLightsZ, frontLightZ;

      const TGLOrthoCamera* orthoCamera = dynamic_cast<const TGLOrthoCamera*>(&camera);
      if (orthoCamera) {
         // Find distance from near clip plane to furstum center - i.e. vector of half
         // clip depth. Ortho lights placed this distance from eye point
         sideLightsZ =
            camera.FrustumPlane(TGLCamera::kNear).DistanceTo(camera.FrustumCenter())*0.7;
         frontLightZ = sideLightsZ;
      } else {
         // Perspective camera
         // Extract vector from camera eye point to center.
         // Camera must have been applied already.
         TGLVector3 eyeVector = camera.EyePoint() - camera.FrustumCenter();

         // Pull forward slightly (0.85) to avoid to sharp a cutoff
         sideLightsZ = eyeVector.Mag() * -0.85;
         frontLightZ = 0.2 * lightRadius;
      }

      // Reset the modelview so static lights are placed in fixed eye space
      // This will destroy camera application.
      glLoadIdentity();

      // 0: Front, 1: Top, 2: Bottom, 3: Left, 4: Right
      TGLVertex3 c = bbox.Center();
      TGLVector3 center(c.X(), c.Y(), c.Z());
      camera.RefModelViewMatrix().MultiplyIP(center);
      // Float_t pos0[] = { center.X(), center.Y(), frontLightZ, 1.0 };
      Float_t pos0[] = { 0.0,        0.0,                      frontLightZ, 1.0 };
      Float_t pos1[] = { center.X(), center.Y() + lightRadius, sideLightsZ, 1.0 };
      Float_t pos2[] = { center.X(), center.Y() - lightRadius, sideLightsZ, 1.0 };
      Float_t pos3[] = { center.X() - lightRadius, center.Y(), sideLightsZ, 1.0 };
      Float_t pos4[] = { center.X() + lightRadius, center.Y(), sideLightsZ, 1.0 };

      Float_t specular = fUseSpecular ? fSpecularPower : 0.0f;
      const Float_t frontLightColor[] = { fFrontPower, fFrontPower, fFrontPower, 1.0f };
      const Float_t sideLightColor[]  = { fSidePower,  fSidePower,  fSidePower,  1.0f };
      const Float_t specLightColor[]  = { specular,    specular,    specular,    1.0f };

      glLightfv(GL_LIGHT0, GL_POSITION, pos0);
      glLightfv(GL_LIGHT0, GL_DIFFUSE,  frontLightColor);
      glLightfv(GL_LIGHT0, GL_SPECULAR, specLightColor);

      glLightfv(GL_LIGHT1, GL_POSITION, pos1);
      glLightfv(GL_LIGHT1, GL_DIFFUSE,  sideLightColor);
      glLightfv(GL_LIGHT2, GL_POSITION, pos2);
      glLightfv(GL_LIGHT2, GL_DIFFUSE,  sideLightColor);
      glLightfv(GL_LIGHT3, GL_POSITION, pos3);
      glLightfv(GL_LIGHT3, GL_DIFFUSE,  sideLightColor);
      glLightfv(GL_LIGHT4, GL_POSITION, pos4);
      glLightfv(GL_LIGHT4, GL_DIFFUSE,  sideLightColor);
   }

   // Set light states everytime - must be defered until now when we know we
   // are in the correct thread for GL context
   // TODO: Could detect state change and only adjust if a change
   for (UInt_t light = 0; (1<<light) < kLightMask; light++)
   {
      if ((1<<light) & fLightState)
      {
         glEnable(GLenum(GL_LIGHT0 + light));

         // Debug mode - show active lights in yellow
         if (debug)
         {
            // Lighting itself needs to be disable so a single one can show...!
            glDisable(GL_LIGHTING);
            Float_t position[4]; // Only float parameters for lights (no double)....
            glGetLightfv(GLenum(GL_LIGHT0 + light), GL_POSITION, position);
            Double_t size = bbox.Extents().Mag() / 10.0;
            TGLVertex3 dPosition(position[0], position[1], position[2]);
            TGLUtil::DrawSphere(dPosition, size, TGLUtil::fgYellow);
            glEnable(GL_LIGHTING);
         }
      }
      else
      {
         glDisable(GLenum(GL_LIGHT0 + light));
      }
   }

   // Restore camera which was applied before we were called, and is disturbed
   // by static light positioning above.
   glPopMatrix();
}
