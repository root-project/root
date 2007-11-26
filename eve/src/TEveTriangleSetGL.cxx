// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveTriangleSetGL.h>
#include <TEveTriangleSet.h>
#include <TVector3.h>

#include <TGLIncludes.h>

//______________________________________________________________________________
// TEveTriangleSetGL
//
// GL-renderer for TEveTriangleSet class.
//
// See also: TGLObject, TGLLogicalShape.

ClassImp(TEveTriangleSetGL)

//______________________________________________________________________________
TEveTriangleSetGL::TEveTriangleSetGL() : TGLObject(), fM(0)
{
   // Constructor.

   // fDLCache = false; // Disable display list.
}

//______________________________________________________________________________
TEveTriangleSetGL::~TEveTriangleSetGL()
{
   // Destructor.
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveTriangleSetGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if(SetModelCheckClass(obj, TEveTriangleSet::Class())) {
      fM = dynamic_cast<TEveTriangleSet*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveTriangleSetGL::SetBBox()
{
   // Set bounding-box from the model.

   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveTriangleSet*)fExternalObj)->AssertBBox());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTriangleSetGL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
{
   // Low-level GL rendering.

   TEveTriangleSet& TS = *fM;
   Bool_t isScaled = TS.fHMTrans.IsScale();

   GLint ex_shade_model;
   glGetIntegerv(GL_SHADE_MODEL, &ex_shade_model);
   glShadeModel(GL_FLAT);

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

   glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
   glEnable(GL_COLOR_MATERIAL);
   glPolygonMode(GL_FRONT, GL_FILL);
   glPolygonMode(GL_BACK,  GL_LINE);
   glDisable(GL_CULL_FACE);
   if (isScaled) glEnable(GL_NORMALIZE);
   glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
   glVertexPointer(3, GL_FLOAT, 0, TS.fVerts);
   glEnableClientState(GL_VERTEX_ARRAY);

   Int_t*   T = TS.fTrings;
   Float_t* N = TS.fTringNorms;
   UChar_t* C = TS.fTringCols;

   TVector3 e1, e2, n;

   glBegin(GL_TRIANGLES);
   for(Int_t t=0; t<TS.fNTrings; ++t) {
      if (N) {
         glNormal3fv(N); N += 3;
      } else {
         Float_t* v0 = TS.Vertex(T[0]);
         Float_t* v1 = TS.Vertex(T[1]);
         Float_t* v2 = TS.Vertex(T[2]);
         e1.SetXYZ(v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]);
         e2.SetXYZ(v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]);
         n = e1.Cross(e2);
         if (!isScaled) n.SetMag(1);
         glNormal3d(n.x(), n.y(), n.z());
      }
      if (C) {
         glColor3ubv(C);  C += 3;
      }
      glArrayElement(T[0]);
      glArrayElement(T[1]);
      glArrayElement(T[2]);
      T += 3;
   }
   glEnd();

   glPopClientAttrib();
   glPopAttrib();
   glShadeModel(ex_shade_model);
}
