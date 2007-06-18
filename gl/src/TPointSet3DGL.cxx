// @(#)root/gl:$Name:  $:$Id: TPointSet3DGL.cxx,v 1.8 2007/06/11 19:56:34 brun Exp $
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef WIN32
#include "Windows4root.h"
#endif

#include "TPointSet3DGL.h"
#include "TPointSet3D.h"

#include <TGLRnrCtx.h>
#include <TGLSelectRecord.h>
#include <TGLIncludes.h>

//______________________________________________________________________
// TPointSet3DGL
//
// Direct OpenGL renderer for TPointSet3D.

ClassImp(TPointSet3DGL)

//______________________________________________________________________________
Bool_t TPointSet3DGL::SetModel(TObject* obj)
{
   // Set model.

   return SetModelCheckClass(obj, TPointSet3D::Class());
}

//______________________________________________________________________________
void TPointSet3DGL::SetBBox()
{
   // Set bounding-box.

   SetAxisAlignedBBox(((TPointSet3D*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
Bool_t TPointSet3DGL::ShouldDLCache(const TGLRnrCtx & rnrCtx) const
{
   // Override from TGLDrawable.
   // To account for large point-sizes we modify the projection matrix
   // during selection and thus we need a direct draw.

   if (rnrCtx.Selection()) return kFALSE;
   return fDLCache;
}

//______________________________________________________________________________
void TPointSet3DGL::Draw(TGLRnrCtx & rnrCtx) const
{
   // Draw function for TPointSet3D. Skips line-pass of outline mode.

   if (rnrCtx.DrawPass() == TGLRnrCtx::kPassOutlineLine)
      return;

   TGLObject::Draw(rnrCtx);
}

//______________________________________________________________________________
void TPointSet3DGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Direct GL rendering for TPointSet3D.

   //printf("TPointSet3DGL::DirectDraw Style %d, LOD %d\n", rnrCtx.Style(), rnrCtx.LOD());
   //printf("  sel=%d, secsel=%d\n", rnrCtx.Selection(), rnrCtx.SecSelection());

   TPointSet3D& q = * (TPointSet3D*) fExternalObj;

   if (q.Size() <= 0) return;

   glPushAttrib(GL_POINT_BIT | GL_LINE_BIT | GL_ENABLE_BIT);

   glDisable(GL_LIGHTING);

   Int_t ms = q.GetMarkerStyle();
   if (ms != 2 && ms != 3 && ms != 5 && ms != 28)
      RenderPoints(rnrCtx);
   else
      RenderCrosses(rnrCtx);


   glPopAttrib();
}

//______________________________________________________________________________
void TPointSet3DGL::RenderPoints(TGLRnrCtx & rnrCtx) const
{
   // Render markers as circular or square points.

   TPointSet3D& q = * (TPointSet3D*) fExternalObj;

   Float_t size = 5*q.GetMarkerSize();
   Int_t  style = q.GetMarkerStyle();
   if (style == 4 || style == 20 || style == 24) {
      if (style == 4 || style == 24)
         glEnable(GL_BLEND);
      glEnable(GL_POINT_SMOOTH);
   } else {
      glDisable(GL_POINT_SMOOTH);
      if      (style == 1) size = 1;
      else if (style == 6) size = 2;
      else if (style == 7) size = 3;
   }
   glPointSize(size);

   // During selection extend picking region for large point-sizes.
   static const Int_t sPickRadius = 3; // Hardcoded also in TGLViewer::RequestSelect()
   Bool_t changePM = kFALSE;
   if (rnrCtx.Selection() && size > sPickRadius) {
      changePM = kTRUE;
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      Float_t pm[16];
      glGetFloatv(GL_PROJECTION_MATRIX, pm);
      Float_t scale = (Float_t) sPickRadius / size;
      for (Int_t i=0; i<=12; i+=4) {
         pm[i] *= scale; pm[i+1] *= scale;
      }
      glLoadMatrixf(pm);
   }

   if (rnrCtx.SecSelection()) {

      const Float_t* p = q.GetP();
      const Int_t    n = q.Size();
      glPushName(0);
      for (Int_t i=0; i<n; ++i, p+=3) {
         glLoadName(i);
         glBegin(GL_POINTS);
         glVertex3fv(p);
         glEnd();
      }
      glPopName();

   } else {

      glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
      glVertexPointer(3, GL_FLOAT, 0, q.GetP());
      glEnableClientState(GL_VERTEX_ARRAY);

      { // Circumvent bug in ATI's linux drivers.
         Int_t nleft = q.Size();
         Int_t ndone = 0;
         const Int_t maxChunk = 8192;
         while (nleft > maxChunk) {
            glDrawArrays(GL_POINTS, ndone, maxChunk);
            nleft -= maxChunk;
            ndone += maxChunk;
         }
         glDrawArrays(GL_POINTS, ndone, nleft);
      }

      glPopClientAttrib();

   }

   if (changePM) {
      glPopMatrix();
      glMatrixMode(GL_MODELVIEW);
   }
}

//______________________________________________________________________________
void TPointSet3DGL::RenderCrosses(TGLRnrCtx & rnrCtx) const
{
   // Render markers as crosses.

   TPointSet3D& q = * (TPointSet3D*) fExternalObj;

   const Float_t* p = q.GetP();
   const Float_t  d = 5*q.GetMarkerSize();
   const Int_t    n = q.Size();

   if (q.GetMarkerStyle() == 28) {
      glEnable(GL_BLEND);
      glEnable(GL_LINE_SMOOTH);
      glLineWidth(2);
   } else {
      glDisable(GL_LINE_SMOOTH);
   }

   if (rnrCtx.SecSelection()) {

      glPushName(0);
      for (Int_t i=0; i<n; ++i, p+=3) {
         glLoadName(i);
         glBegin(GL_LINES);
         glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2]);
         glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2]);
         glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d);
         glEnd();
      }
      glPopName();

   } else {

      glBegin(GL_LINES);
      for (Int_t i=0; i<n; ++i, p+=3) {
         glVertex3f(p[0]-d, p[1], p[2]); glVertex3f(p[0]+d, p[1], p[2]);
         glVertex3f(p[0], p[1]-d, p[2]); glVertex3f(p[0], p[1]+d, p[2]);
         glVertex3f(p[0], p[1], p[2]-d); glVertex3f(p[0], p[1], p[2]+d);
      }
      glEnd();

   }
}

//______________________________________________________________________________
void TPointSet3DGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes secondary selection from TGLViewer.
   // Calls TPointSet3D::PointSelected(Int_t) with index of selected
   // point as an argument.

   if (rec.GetN() < 2) return;
   TPointSet3D& q = * (TPointSet3D*) fExternalObj;
   q.PointSelected(rec.GetItem(1));
}
