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
#include "TFTGLManager.h"
#include "TGLUtil.h"
#include "TGLCamera.h"

#include "FTFont.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"
#include "TGLBoundingBox.h"

//______________________________________________________________________________
// TEveTextGL
//
// OpenGL renderer class for TEveText.
//

ClassImp(TEveTextGL);

//______________________________________________________________________________
TEveTextGL::TEveTextGL() :
   TGLObject(),
   fSize(0), fFile(0), fMode(0),
   fFont(0), fM(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
Bool_t TEveTextGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (SetModelCheckClass(obj, TEveText::Class())) {
      fM = dynamic_cast<TEveText*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveTextGL::SetBBox()
{
   // Set bounding box.

   fBoundingBox.SetEmpty();
}

//______________________________________________________________________________
void TEveTextGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   static const TEveException eH("TEveTextGL::DirectDraw ");

   if (fSize != fM->GetSize() || fFile != fM->GetFile() || fMode != fM->GetMode())
   {
      if (fFont)
         rnrCtx.ReleaseFont(fSize, fFile, fMode);

      fSize = fM->GetSize();
      fFile = fM->GetFile();
      fMode = fM->GetMode();
      fFont = rnrCtx.GetFont(fSize, fFile, fMode);
   }

   //  bbox initialisation
   if (fBoundingBox.IsEmpty() &&
       fMode != TFTGLManager::kBitmap && fMode != TFTGLManager::kPixmap)
   {
      Float_t bbox[6];
      fFont->BBox(fM->GetText(), bbox[0], bbox[1], bbox[2],
                                 bbox[3], bbox[4], bbox[5]);

      if (fMode == TFTGLManager::kExtrude) {
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
   TGLCapabilitySwitch lights(GL_LIGHTING, fM->GetLighting());
   switch(fMode)
   {
      case TFTGLManager::kBitmap:
      case TFTGLManager::kPixmap:
      {
         if (rnrCtx.Selection())
         {
            // calculate coordinates in 3D
            const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
            GLdouble mm[16];
            GLint    vp[4];
            glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
            glGetIntegerv(GL_VIEWPORT, vp);

            fX[0][0] = fX[0][1] = fX[0][2] = 0;
            GLdouble x, y, z;
            gluProject(fX[0][0], fX[0][1], fX[0][2], mm, pm, vp, &x, &y, &z);
            Float_t bbox[6];
            fFont->BBox(fM->GetText(), bbox[0], bbox[1], bbox[2],
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
         }
         else
         {
            TGLCapabilitySwitch blending(GL_BLEND, kTRUE);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glAlphaFunc(GL_GEQUAL, 0.0625);
            glEnable(GL_ALPHA_TEST);

            glRasterPos3i(0, 0, 0);
            fFont->Render(fM->GetText());
         }
         break;
      }
      case TFTGLManager::kOutline:
      case TFTGLManager::kExtrude:
      case TFTGLManager::kPolygon:
      {
         TGLCapabilitySwitch normalize(GL_NORMALIZE, kTRUE);
         glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
         TGLCapabilitySwitch col(GL_COLOR_MATERIAL, kTRUE);
         TGLCapabilitySwitch cull(GL_CULL_FACE, kFALSE);
         if (fM->GetExtrude() != 1.0)
         {
            glPushMatrix();
            glScalef(1.0f, 1.0f, fM->GetExtrude());
            fFont->Render(fM->GetText());
            glPopMatrix();
         }
         else
         {
            fFont->Render(fM->GetText());
         }
         break;
      }
      case TFTGLManager::kTexture:
      {
         TGLCapabilitySwitch alpha(GL_ALPHA_TEST, kTRUE);
         TGLCapabilitySwitch blending(GL_BLEND, kTRUE);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

         glAlphaFunc(GL_GEQUAL, 0.0625);
         glEnable(GL_ALPHA_TEST);

         glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT);
         TGLCapabilitySwitch texture(GL_TEXTURE_2D, kTRUE);
         TGLCapabilitySwitch col(GL_COLOR_MATERIAL, kTRUE);
         fFont->Render(fM->GetText());
         glPopAttrib();
         break;
      }
      default:
      {
         throw(eH + "unsupported FTGL-type.");
      }
   }
}
