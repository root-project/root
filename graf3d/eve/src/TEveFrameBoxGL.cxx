// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveFrameBoxGL.h"
#include "TEveFrameBox.h"

#include "TGLIncludes.h"
#include "TGLUtil.h"

#include "TMath.h"

//______________________________________________________________________________
//
// A class encapsulating GL rendering of TEveFrameBox via a static
// meber function.

ClassImp(TEveFrameBoxGL);

//______________________________________________________________________________
void TEveFrameBoxGL::RenderFrame(const TEveFrameBox& b, Bool_t fillp)
{
   // Render the frame with GL.

   const Float_t * p = b.fFramePoints;
   Float_t normal[3];

   if (b.fFrameType == TEveFrameBox::kFT_Quad)
   {
      if (fillp)
      {
         TMath::Normal2Plane(p, p+3, p+6, normal);
         glNormal3fv(normal);
         glBegin(GL_POLYGON);
      }
      else
      {
         glBegin(GL_LINE_LOOP);
      }
      Int_t nPoints = b.fFrameSize / 3;
      for (Int_t i = 0; i < nPoints; ++i, p += 3)
         glVertex3fv(p);
      glEnd();
   }
   else if (b.fFrameType == TEveFrameBox::kFT_Box)
   {
      if (fillp)
      {
         glBegin(GL_QUADS);
         TMath::Normal2Plane(p, p+3, p+6, normal);
         glNormal3fv(normal);
         glVertex3fv(p);      glVertex3fv(p + 3);
         glVertex3fv(p + 6);  glVertex3fv(p + 9);
         // top:    7654
         TMath::Normal2Plane(p+21, p+18, p+15, normal);
         glNormal3fv(normal);
         glVertex3fv(p + 21); glVertex3fv(p + 18);
         glVertex3fv(p + 15); glVertex3fv(p + 12);
         // back:  0451
         TMath::Normal2Plane(p, p+12, p+15, normal);
         glNormal3fv(normal);
         glVertex3fv(p);      glVertex3fv(p + 12);
         glVertex3fv(p + 15); glVertex3fv(p + 3);
         // front:   3267
         TMath::Normal2Plane(p+9, p+6, p+18, normal);
         glNormal3fv(normal);
         glVertex3fv(p + 9);   glVertex3fv(p + 6);
         glVertex3fv(p + 18);  glVertex3fv(p + 21);
         // left:    0374
         TMath::Normal2Plane(p, p+9, p+21, normal);
         glNormal3fv(normal);
         glVertex3fv(p);       glVertex3fv(p + 9);
         glVertex3fv(p + 21);  glVertex3fv(p + 12);
         // right:   1562
         TMath::Normal2Plane(p+3, p+15, p+18, normal);
         glNormal3fv(normal);
         glVertex3fv(p + 3);   glVertex3fv(p + 15);
         glVertex3fv(p + 18);  glVertex3fv(p + 6);
         glEnd();
      }
      else
      {
         glBegin(GL_LINE_STRIP);
         glVertex3fv(p);       glVertex3fv(p + 3);
         glVertex3fv(p + 6);   glVertex3fv(p + 9);
         glVertex3fv(p);
         glVertex3fv(p + 12);  glVertex3fv(p + 15);
         glVertex3fv(p + 18);  glVertex3fv(p + 21);
         glVertex3fv(p + 12);
         glEnd();
         glBegin(GL_LINES);
         glVertex3fv(p + 3);   glVertex3fv(p + 15);
         glVertex3fv(p + 6);   glVertex3fv(p + 18);
         glVertex3fv(p + 9);   glVertex3fv(p + 21);
         glEnd();
      }
   }
}

//______________________________________________________________________________
void TEveFrameBoxGL::Render(const TEveFrameBox* box)
{
   // Render the frame-box with GL.

   const TEveFrameBox& b = *box;

   glPushAttrib(GL_POLYGON_BIT | GL_LINE_BIT | GL_ENABLE_BIT);

   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glDisable(GL_CULL_FACE);

   if (b.fDrawBack)
   {
      GLboolean lmts;
      glGetBooleanv(GL_LIGHT_MODEL_TWO_SIDE, &lmts);
      if (!lmts) glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(2, 2);
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      if (b.fBackRGBA[3] < 255)
      {
         glEnable(GL_BLEND);
         glDepthMask(GL_FALSE);
      }

      TGLUtil::Color4ubv(b.fBackRGBA);
      RenderFrame(b, kTRUE);

      if (!lmts) glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
   }

   glDisable(GL_LIGHTING);
   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glEnable(GL_LINE_SMOOTH);

   TGLUtil::LineWidth(b.fFrameWidth);
   TGLUtil::Color4ubv(b.fFrameRGBA);
   RenderFrame(b, b.fFrameFill);

   glPopAttrib();
}
