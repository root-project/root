// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBoxGL.h"
#include "TEveBox.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

#include "TMath.h"

//______________________________________________________________________________
// OpenGL renderer class for TEveBox.
//

ClassImp(TEveBoxGL);

//______________________________________________________________________________
TEveBoxGL::TEveBoxGL() :
   TGLObject(), fM(0)
{
   // Constructor.

   // fDLCache = kFALSE; // Disable display list.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveBoxGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (SetModelCheckClass(obj, TEveBox::Class())) {
      fM = dynamic_cast<TEveBox*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveBoxGL::SetBBox()
{
   // Set bounding box.

   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveBox*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

namespace
{
   void subtract_and_normalize(const Float_t a[3], const Float_t b[3],
                               Float_t o[3])
   {
      // Calculate a - b and normalize the result.
      o[0] = a[0] - b[0];
      o[1] = a[1] - b[1];
      o[2] = a[2] - b[2];
      Float_t d = sqrtf(o[0]*o[0] + o[1]*o[1] + o[2]*o[2]);
      if (d != 0)
      {
         d = 1.0f / d;
         o[0] *= d;
         o[1] *= d;
         o[2] *= d;
      }
   }
}

//______________________________________________________________________________
void TEveBoxGL::RenderOutline(const Float_t p[8][3]) const
{
   // Render box with without normals.
   // To be used with lightning off, for outline.

   glBegin(GL_LINE_STRIP);
   glVertex3fv(p[0]); glVertex3fv(p[1]);
   glVertex3fv(p[5]); glVertex3fv(p[6]);
   glVertex3fv(p[2]); glVertex3fv(p[3]);
   glVertex3fv(p[7]); glVertex3fv(p[4]);
   glVertex3fv(p[0]); glVertex3fv(p[3]);
   glEnd();

   glBegin(GL_LINES);
   glVertex3fv(p[1]); glVertex3fv(p[2]);
   glVertex3fv(p[4]); glVertex3fv(p[5]);
   glVertex3fv(p[6]); glVertex3fv(p[7]);
   glEnd();
}

//______________________________________________________________________________
void TEveBoxGL::RenderBoxStdNorm(const Float_t p[8][3]) const
{
   // Render box with "standard" normals.

   glBegin(GL_QUADS);

   // bottom: 0123
   glNormal3f(0, 0, -1);
   glVertex3fv(p[0]);  glVertex3fv(p[1]);
   glVertex3fv(p[2]);  glVertex3fv(p[3]);
   // top:    7654
   glNormal3f(0, 0, 1);
   glVertex3fv(p[7]); glVertex3fv(p[6]);
   glVertex3fv(p[5]); glVertex3fv(p[4]);
   // back:  0451
   glNormal3f(0, 1, 0);
   glVertex3fv(p[0]); glVertex3fv(p[4]);
   glVertex3fv(p[5]); glVertex3fv(p[1]);
   // front:   3267
   glNormal3f(0, -1, 0);
   glVertex3fv(p[3]); glVertex3fv(p[2]);
   glVertex3fv(p[6]); glVertex3fv(p[7]);
   // left:    0374
   glNormal3f(-1, 0, 0);
   glVertex3fv(p[0]); glVertex3fv(p[3]);
   glVertex3fv(p[7]); glVertex3fv(p[4]);
   // right:   1562
   glNormal3f(1, 0, 0);
   glVertex3fv(p[1]); glVertex3fv(p[5]);
   glVertex3fv(p[6]); glVertex3fv(p[2]);

   glEnd();
}

//______________________________________________________________________________
void TEveBoxGL::RenderBoxAutoNorm(const Float_t p[8][3]) const
{
   // Render box, calculate normals on the fly from first three points.

   Float_t e[6][3], n[3];
   subtract_and_normalize(p[1], p[0], e[0]);
   subtract_and_normalize(p[3], p[0], e[1]);
   subtract_and_normalize(p[4], p[0], e[2]);
   subtract_and_normalize(p[5], p[6], e[3]);
   subtract_and_normalize(p[4], p[6], e[4]);
   subtract_and_normalize(p[2], p[6], e[5]);

   glBegin(GL_QUADS);

   // bottom: 0123
   glNormal3fv(TMath::Cross(e[0], e[1], n));
   glVertex3fv(p[0]); glVertex3fv(p[1]);
   glVertex3fv(p[2]); glVertex3fv(p[3]);
   // top:    7654
   glNormal3fv(TMath::Cross(e[3], e[4], n));
   glVertex3fv(p[7]); glVertex3fv(p[6]);
   glVertex3fv(p[5]); glVertex3fv(p[4]);
   // back:  0451
   glNormal3fv(TMath::Cross(e[2], e[1], n));
   glVertex3fv(p[0]); glVertex3fv(p[4]);
   glVertex3fv(p[5]); glVertex3fv(p[1]);
   // front:   3267
   glNormal3fv(TMath::Cross(e[4], e[5], n));
   glVertex3fv(p[3]); glVertex3fv(p[2]);
   glVertex3fv(p[6]); glVertex3fv(p[7]);
   // left:    0374
   glNormal3fv(TMath::Cross(e[1], e[2], n));
   glVertex3fv(p[0]); glVertex3fv(p[3]);
   glVertex3fv(p[7]); glVertex3fv(p[4]);
   // right:   1562
   glNormal3fv(TMath::Cross(e[5], e[3], n));
   glVertex3fv(p[1]); glVertex3fv(p[5]);
   glVertex3fv(p[6]); glVertex3fv(p[2]);

   glEnd();
}

//______________________________________________________________________________
void TEveBoxGL::Draw(TGLRnrCtx& rnrCtx) const
{
   // Render with OpenGL.

   if (rnrCtx.IsDrawPassOutlineLine())
      return;

   if (fM->fHighlightFrame && rnrCtx.Highlight())
   {
      if (fM->fDrawFrame)
      {
         TGLUtil::LineWidth(fM->fLineWidth);
         TGLUtil::Color(fM->fLineColor);
      }
      RenderOutline(fM->fVertices);
   }
   else
   {
      TGLObject::Draw(rnrCtx);
   }
}

//______________________________________________________________________________
void TEveBoxGL::DirectDraw(TGLRnrCtx&) const
{
   // Render with OpenGL, create display-list.

   fMultiColor = (fM->fDrawFrame && fM->fFillColor != fM->fLineColor);

   glPushAttrib(GL_ENABLE_BIT);

   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.0f, 1.0f);
   RenderBoxAutoNorm(fM->fVertices);
   glDisable(GL_POLYGON_OFFSET_FILL);

   // Frame
   if (fM->fDrawFrame)
   {
      TGLUtil::Color(fM->fLineColor);
      TGLUtil::LineWidth(fM->fLineWidth);
      glEnable(GL_LINE_SMOOTH);
      RenderOutline(fM->fVertices);
   }

   glPopAttrib();
}
