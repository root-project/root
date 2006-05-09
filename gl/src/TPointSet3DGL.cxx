// @(#)root/gl:$Name:  $:$Id: TPointSet3DGL.cxx,v 1.5 2006/04/12 15:49:07 brun Exp $
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

#include <GL/gl.h>

//______________________________________________________________________
// TPointSet3DGL
//
// Direct OpenGL renderer for TPointSet3D.

ClassImp(TPointSet3DGL)

//______________________________________________________________________________
TPointSet3DGL::TPointSet3DGL() : TGLObject()
{}

//______________________________________________________________________________
Bool_t TPointSet3DGL::SetModel(TObject* obj)
{
   // Set model.

   return SetModelCheckClass(obj, "TPointSet3D");
}

//______________________________________________________________________________
void TPointSet3DGL::SetBBox()
{
   // Set bounding-box.

   SetAxisAlignedBBox(((TPointSet3D*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TPointSet3DGL::DirectDraw(const TGLDrawFlags & /*flags*/) const
{
   // Direct GL rendering for TPointSet3D.

   // printf("TPointSet3DGL::DirectDraw Style %d, LOD %d\n", flags.Style(), flags.LOD());

   TPointSet3D& q = * (TPointSet3D*) fExternalObj;

   if (q.GetN() <= 0) return;

   Int_t qms = q.GetMarkerStyle();

   glPushAttrib(GL_POINT_BIT | GL_ENABLE_BIT);

   glDisable(GL_LIGHTING);

   if (qms == 20 || qms == 21) {  // 20 ~ full scalable circle; 21 ~ fs square
      glEnable(GL_BLEND);
      glPointSize(q.GetMarkerSize());
   }
   if (q.GetMarkerStyle() == 20) {
      glEnable(GL_POINT_SMOOTH);
   } else {
      glDisable(GL_POINT_SMOOTH);
   }

   glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
   glVertexPointer(3, GL_FLOAT, 0, q.GetP());
   glEnableClientState(GL_VERTEX_ARRAY);

   glDrawArrays(GL_POINTS, 0, q.GetN());

   glPopClientAttrib();

   glPopAttrib();
}
