// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTextGL.h"
#include "TEveText.h"
#include "TGLUtil.h"
#include "TGLCamera.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"
#include "TGLBoundingBox.h"

//______________________________________________________________________________
//
// OpenGL renderer class for TEveText.
//

ClassImp(TEveTextGL);

//______________________________________________________________________________
TEveTextGL::TEveTextGL() :
   TGLObject(),
   fM(0),
   fFont()
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveTextGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveText>(obj);
   return kTRUE;
}

//______________________________________________________________________________
void TEveTextGL::SetBBox()
{
   // Set bounding box.

   fBoundingBox.SetEmpty();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTextGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   static const TEveException eH("TEveTextGL::DirectDraw ");

   Int_t fm = fM->GetFontMode();
   if (fm == TGLFont::kBitmap || fm == TGLFont::kPixmap || fm == TGLFont::kTexture)
      rnrCtx.RegisterFont(fM->GetFontSize(), fM->GetFontFile(), fM->GetFontMode(), fFont);
   else
      rnrCtx.RegisterFontNoScale(fM->GetFontSize(), fM->GetFontFile(), fM->GetFontMode(), fFont);

   fFont.SetDepth(fM->GetExtrude());

   //  bbox initialisation
   if (fBoundingBox.IsEmpty() && fFont.GetMode() > TGLFont::kPixmap)
   {
      Float_t bbox[6];
      fFont.BBox(fM->GetText(), bbox[0], bbox[1], bbox[2],
                 bbox[3], bbox[4], bbox[5]);

      if (fFont.GetMode() == TGLFont::kExtrude) {
         // Depth goes, the other z-way, swap.
         Float_t tmp = bbox[2];
         bbox[2] = bbox[5] * fM->GetExtrude();
         bbox[5] = tmp     * fM->GetExtrude();
      } else {
         bbox[2] = -0.005*(bbox[4] - bbox[1]);
         bbox[5] = -0.005*(bbox[4] - bbox[1]);
      }

      TGLVertex3 low (bbox[0], bbox[1], bbox[2]);
      TGLVertex3 high(bbox[3], bbox[4], bbox[5]);

      TEveTextGL* ncthis = const_cast<TEveTextGL*>(this);
      ncthis->fBoundingBox.SetAligned(low, high);
      ncthis->UpdateBoundingBoxesOfPhysicals();
   }

   // rendering
   glPushMatrix();
   fFont.PreRender(fM->GetAutoLighting(), fM->GetLighting());
   switch (fFont.GetMode())
   {
      case TGLFont::kBitmap:
      case TGLFont::kPixmap:
         if (rnrCtx.Selection()) {
            // calculate 3D coordinates for picking
            const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
            GLdouble mm[16];
            GLint    vp[4];
            glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
            glGetIntegerv(GL_VIEWPORT, vp);

            fX[0][0] = fX[0][1] = fX[0][2] = 0;
            GLdouble x, y, z;
            gluProject(fX[0][0], fX[0][1], fX[0][2], mm, pm, vp, &x, &y, &z);
            Float_t bbox[6];
            fFont.BBox(fM->GetText(), bbox[0], bbox[1], bbox[2],
                       bbox[3], bbox[4], bbox[5]);
            gluUnProject(x + bbox[0], y + bbox[1], z, mm, pm, vp, &fX[0][0], &fX[0][1], &fX[0][2]);
            gluUnProject(x + bbox[3], y + bbox[1], z, mm, pm, vp, &fX[1][0], &fX[1][1], &fX[1][2]);
            gluUnProject(x + bbox[3], y + bbox[4], z, mm, pm, vp, &fX[2][0], &fX[2][1], &fX[2][2]);
            gluUnProject(x + bbox[0], y + bbox[4], z, mm, pm, vp, &fX[3][0], &fX[3][1], &fX[3][2]);

            glBegin(GL_POLYGON);
            glVertex3dv(fX[0]);
            glVertex3dv(fX[1]);
            glVertex3dv(fX[2]);
            glVertex3dv(fX[3]);
            glEnd();
         } else {
            glRasterPos3i(0, 0, 0);
            fFont.Render(fM->GetText());
         }
         break;
      case TGLFont::kOutline:
      case TGLFont::kExtrude:
      case TGLFont::kPolygon:
         glPolygonOffset(fM->GetPolygonOffset(0), fM->GetPolygonOffset(1));
         if (fM->GetExtrude() != 1.0) {
            glPushMatrix();
            glScalef(1.0f, 1.0f, fM->GetExtrude());
            fFont.Render(fM->GetText());
            glPopMatrix();
         } else {
            fFont.Render(fM->GetText());
         }
         break;
      case TGLFont::kTexture:
         glPolygonOffset(fM->GetPolygonOffset(0), fM->GetPolygonOffset(1));
         fFont.Render(fM->GetText());
         break;
      default:
         throw(eH + "unsupported FTGL-type.");
   }
   fFont.PostRender();
   glPopMatrix();
}
