// @(#)root/gl:$Name:  $:$Id: TGLPolyMarker.cxx,v 1.4 2006/12/09 23:02:27 rdm Exp $
// Author:  Timur Pocheptsov  03/08/2004
// NOTE: This code moved from obsoleted TGLSceneObject.h / .cxx - see these
// attic files for previous CVS history

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TGLPolyMarker.h"
#include "TGLDrawFlags.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

#include "TAttMarker.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

ClassImp(TGLPolyMarker)

//______________________________________________________________________________
TGLPolyMarker::TGLPolyMarker(const TBuffer3D & buffer) :
   TGLLogicalShape(buffer),
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fStyle(7),
   fSize(1.)
{
   //TAttMarker is not TObject descendant, so I need dynamic_cast
   if (TAttMarker *realObj = dynamic_cast<TAttMarker *>(buffer.fID)) {
      fStyle = realObj->GetMarkerStyle();
      fSize  = realObj->GetMarkerSize() / 2.;
   }
}

//______________________________________________________________________________
void TGLPolyMarker::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyMarker::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   const Double_t *vertices = &fVertices[0];
   UInt_t size = fVertices.size();
   Int_t stacks = 6, slices = 6;
   Float_t pixelSize  = 1;
   Double_t topRadius = fSize;

   switch (fStyle) {
   case 27:
      stacks = 2, slices = 4;
   case 4:case 8:case 20:case 24:
      for (UInt_t i = 0; i < size; i += 3) {
         glPushMatrix();
         glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
         gluSphere(fgQuad.Get(), fSize, slices, stacks);
         glPopMatrix();
      }
      break;
   case 22:case 26:
      topRadius = 0.;
   case 21:case 25:
      for (UInt_t i = 0; i < size; i += 3) {
         glPushMatrix();
         glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
         gluCylinder(fgQuad.Get(), fSize, topRadius, fSize, 4, 1);
         glPopMatrix();
      }
      break;
   case 23:
      for (UInt_t i = 0; i < size; i += 3) {
         glPushMatrix();
         glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
         glRotated(180, 1., 0., 0.);
         gluCylinder(fgQuad.Get(), fSize, 0., fSize, 4, 1);
         glPopMatrix();
      }
      break;
   case 3: case 2: case 5:
      DrawStars();
      break;
   case 7:
      pixelSize += 1;
   case 6:
      pixelSize += 1;
   case 1: case 9: case 10: case 11: default:
      glPointSize(pixelSize);
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
      break;
   }
}

//______________________________________________________________________________
void TGLPolyMarker::DrawStars()const
{
   // Draw stars
   glDisable(GL_LIGHTING);
   const Double_t diag = TMath::Sqrt(2 * fSize * fSize) / 2;

   for (UInt_t i = 0; i < fVertices.size(); i += 3) {
      Double_t x = fVertices[i];
      Double_t y = fVertices[i + 1];
      Double_t z = fVertices[i + 2];
      glBegin(GL_LINES);
      if (fStyle == 2 || fStyle == 3) {
         glVertex3d(x - fSize, y, z);
         glVertex3d(x + fSize, y, z);
         glVertex3d(x, y, z - fSize);
         glVertex3d(x, y, z + fSize);
         glVertex3d(x, y - fSize, z);
         glVertex3d(x, y + fSize, z);
      }
      if(fStyle != 2) {
         glVertex3d(x - diag, y - diag, z - diag);
         glVertex3d(x + diag, y + diag, z + diag);
         glVertex3d(x - diag, y - diag, z + diag);
         glVertex3d(x + diag, y + diag, z - diag);
         glVertex3d(x - diag, y + diag, z - diag);
         glVertex3d(x + diag, y - diag, z + diag);
         glVertex3d(x - diag, y + diag, z + diag);
         glVertex3d(x + diag, y - diag, z - diag);
      }
      glEnd();
   }
   glEnable(GL_LIGHTING);
}
