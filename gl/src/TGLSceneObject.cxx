// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.cxx,v 1.28 2005/01/19 13:19:34 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifdef GDK_WIN32
#include "Windows4Root.h"
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include "TAttMarker.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TError.h"

#include "TGLSceneObject.h"
#include "TGLFrustum.h"

#include <assert.h>

#include <iostream> // Remove

ClassImp(TGLSceneObject)

static GLUtriangulatorObj *GetTesselator()
{
   static struct Init {
      Init()
      {
#ifdef GDK_WIN32
         typedef void (CALLBACK *tessfuncptr_t)();
#else
         typedef void (*tessfuncptr_t)();
#endif
         fTess = gluNewTess();

         if (!fTess) {
            Error("GetTesselator::Init", "could not create tesselation object");
         } else {
            gluTessCallback(fTess, (GLenum)GLU_BEGIN, (tessfuncptr_t)glBegin);
            gluTessCallback(fTess, (GLenum)GLU_END, (tessfuncptr_t)glEnd);
            gluTessCallback(fTess, (GLenum)GLU_VERTEX, (tessfuncptr_t)glVertex3dv);
         }
      }
      ~Init()
      {
         if(fTess)
            gluDeleteTess(fTess);
      }
      GLUtriangulatorObj *fTess;
   }singleton;

   return singleton.fTess;
}

static GLUquadric *GetQuadric()
{
   static struct Init {
      Init()
      {
         fQuad = gluNewQuadric();
         if (!fQuad) {
            Error("GetQuadric::Init", "could not create quadric object");
         } else {
            gluQuadricOrientation(fQuad, (GLenum)GLU_OUTSIDE);
            gluQuadricNormals(fQuad, (GLenum)GLU_SMOOTH);
         }
      }
      ~Init()
      {
         if(fQuad)
            gluDeleteQuadric(fQuad);
      }
      GLUquadric *fQuad;
   }singleton;

   return singleton.fQuad;
}

//______________________________________________________________________________
TGLSelection::TGLSelection()
{
   fBBox[0] = fBBox[1] = fBBox[2] =
   fBBox[3] = fBBox[4] = fBBox[5] = 0.;
}

//______________________________________________________________________________
TGLSelection::TGLSelection(const Double_t *bbox)
{
   for (Int_t i= 0; i < 6; ++i) fBBox[i] = bbox[i];
}

//______________________________________________________________________________
TGLSelection::TGLSelection(Double_t xmin, Double_t xmax, Double_t ymin,
                           Double_t ymax, Double_t zmin, Double_t zmax)
{
   fBBox[0] = xmin, fBBox[1] = xmax;
   fBBox[2] = ymin, fBBox[3] = ymax;
   fBBox[4] = zmin, fBBox[5] = zmax;
}

//______________________________________________________________________________
void TGLSelection::DrawBox()const
{
   Double_t xmin = fBBox[0], xmax = fBBox[1];
   Double_t ymin = fBBox[2], ymax = fBBox[3];
   Double_t zmin = fBBox[4], zmax = fBBox[5];

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);

   glColor3d(1., 1., 1.);
   glBegin(GL_LINE_LOOP);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymax, zmin);
   glVertex3d(xmax, ymax, zmin);
   glVertex3d(xmax, ymin, zmin);
   glEnd();
   glBegin(GL_LINE_LOOP);
   glVertex3d(xmin, ymin, zmax);
   glVertex3d(xmin, ymax, zmax);
   glVertex3d(xmax, ymax, zmax);
   glVertex3d(xmax, ymin, zmax);
   glEnd();
   glBegin(GL_LINES);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymin, zmax);
   glVertex3d(xmin, ymax, zmin);
   glVertex3d(xmin, ymax, zmax);
   glVertex3d(xmax, ymax, zmin);
   glVertex3d(xmax, ymax, zmax);
   glVertex3d(xmax, ymin, zmin);
   glVertex3d(xmax, ymin, zmax);
   glEnd();

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
}

//______________________________________________________________________________
void TGLSelection::SetBBox(const Double_t *newBBox)
{
   for (Int_t i= 0; i < 6; ++i) fBBox[i] = newBBox[i];
}

//______________________________________________________________________________
void TGLSelection::SetBBox(Double_t xmin, Double_t xmax, Double_t ymin,
                          Double_t ymax, Double_t zmin, Double_t zmax)
{
   fBBox[0] = xmin, fBBox[1] = xmax;
   fBBox[2] = ymin, fBBox[3] = ymax;
   fBBox[4] = zmin, fBBox[5] = zmax;
}

//______________________________________________________________________________
void TGLSelection::Shift(Double_t x, Double_t y, Double_t z)
{
   fBBox[0] += x, fBBox[1] += x;
   fBBox[2] += y, fBBox[3] += y;
   fBBox[4] += z, fBBox[5] += z;
}

//______________________________________________________________________________
void TGLSelection::Stretch(Double_t xs, Double_t ys, Double_t zs)
{
   Double_t xC = fBBox[0] + (fBBox[1] - fBBox[0]) / 2;
   Double_t yC = fBBox[2] + (fBBox[3] - fBBox[2]) / 2;
   Double_t zC = fBBox[4] + (fBBox[5] - fBBox[4]) / 2;

   Shift(-xC, -yC, -zC);
   fBBox[0] *= xs, fBBox[1] *= xs;
   fBBox[2] *= ys, fBBox[3] *= ys;
   fBBox[4] *= zs, fBBox[5] *= zs;
   Shift(xC, yC, zC);
}

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const TBuffer3D &buffer, const Float_t *color, 
                               UInt_t glName, TObject *obj) :
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()), 
   fColor(),
   fGLName(glName),
   fNextT(0), 
   fRealObject(obj),
   fIsSelected(kFALSE)
{
   SetColor(color, kTRUE);
   fColor[3] = 1.f - buffer.fTransparency / 100.f;
   SetBBox(buffer);
}

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve, 
                               const Float_t *color, UInt_t glName, TObject *obj) :
   fVertices(verticesReserve, 0.), 
   fColor(),
   fGLName(glName),
   fNextT(0),
   fRealObject(obj),
   fIsSelected(kFALSE)
{
   SetColor(color, kTRUE);
   fColor[3] = 1.f - buffer.fTransparency / 100.f;
   SetBBox(buffer);
}

//______________________________________________________________________________
Bool_t TGLSceneObject::IsTransparent()const
{
   return fColor[3] < 1.f;
}

//______________________________________________________________________________
void TGLSceneObject::Shift(Double_t x, Double_t y, Double_t z)
{
   fSelectionBox.Shift(x, y, z);
   for (UInt_t i = 0, e = fVertices.size(); i < e; i += 3) {
      fVertices[i] += x;
      fVertices[i + 1] += y;
      fVertices[i + 2] += z;
   }
}

//______________________________________________________________________________
void TGLSceneObject::Stretch(Double_t xs, Double_t ys, Double_t zs)
{
   fSelectionBox.Stretch(xs, ys, zs);

   const Double_t *bbox = fSelectionBox.GetData();
   Double_t xC = bbox[0] + (bbox[1] - bbox[0]) / 2;
   Double_t yC = bbox[2] + (bbox[3] - bbox[2]) / 2;
   Double_t zC = bbox[4] + (bbox[5] - bbox[4]) / 2;

   Shift(-xC, -yC, -zC);
   for (UInt_t i = 0, e = fVertices.size(); i < e; i += 3) {
      fVertices[i] *= xs;
      fVertices[i + 1] *= ys;
      fVertices[i + 2] *= zs;
   }
   Shift(xC, yC, zC);
}

//______________________________________________________________________________
void TGLSceneObject::SetColor(const Float_t *color, Bool_t fromCtor)
{
   if (!fromCtor) {
      for (Int_t i = 0; i < 17; ++i) fColor[i] = color[i];
   } else {
      if (color) {
         //diffuse and specular
         fColor[0] = color[0];
         fColor[1] = color[1];
         fColor[2] = color[2];
      } else {
         for (Int_t i = 0; i < 12; ++i) fColor[i] = 1.f;
      }
      //ambient
      fColor[4] = fColor[5] = fColor[6] = 0.f;
      //specular
      fColor[8] = fColor[9] = fColor[10] = 0.7f;
      //emission
      fColor[12] = fColor[13] = fColor[14] = 0.f;
      //alpha
      fColor[3] = fColor[7] = fColor[11] = fColor[15] = 1.f;
      //shininess
      fColor[16] = 60.f;
   }
}

//______________________________________________________________________________
void TGLSceneObject::SetBBox(const TBuffer3D & buffer)
{
   // Use the buffer bounding box if provided
   if (buffer.SectionsValid(TBuffer3D::kBoundingBox))  {
      fSelectionBox.SetBBox(buffer.fBBLowVertex[0], buffer.fBBHighVertex[0],
                            buffer.fBBLowVertex[1], buffer.fBBHighVertex[1],
                            buffer.fBBLowVertex[2], buffer.fBBHighVertex[2]);
   }
   // otherwise build a bounding box based on extent of points
   else {
      Double_t xmin = buffer.fPnts[0], xmax = xmin;
      Double_t ymin = buffer.fPnts[1], ymax = ymin;
      Double_t zmin = buffer.fPnts[2], zmax = zmin;

      for (UInt_t nv = 3; nv < buffer.NbPnts(); nv += 3) {
         xmin = TMath::Min(xmin, buffer.fPnts[nv]);
         xmax = TMath::Max(xmax, buffer.fPnts[nv]);
         ymin = TMath::Min(ymin, buffer.fPnts[nv + 1]);
         ymax = TMath::Max(ymax, buffer.fPnts[nv + 1]);
         zmin = TMath::Min(zmin, buffer.fPnts[nv + 2]);
         zmax = TMath::Max(zmax, buffer.fPnts[nv + 2]);
      }

      fSelectionBox.SetBBox(xmin, xmax, ymin, ymax, zmin, zmax);
   }
}

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buff, const Float_t *color, UInt_t glname, TObject *realobj)
               :TGLSceneObject(buff, color, glname, realobj),
                fNormals(3 * buff.NbPols())
{
   fNbPols = buff.NbPols();

   Int_t *segs = buff.fSegs;
   Int_t *pols = buff.fPols;
   Int_t shiftInd = buff.fReflection ? 1 : -1;

   Int_t descSize = 0;

   for (UInt_t i = 0, j = 1; i < fNbPols; ++i, ++j)
   {
      descSize += pols[j] + 1;
      j += pols[j] + 1;
   }

   fPolyDesc.resize(descSize);
   {//fix for scope
   for (UInt_t numPol = 0, currInd = 0, j = 1; numPol < fNbPols; ++numPol) {
      Int_t segmentInd = shiftInd < 0 ? pols[j] + j : j + 1;
      Int_t segmentCol = pols[j];
      Int_t s1 = pols[segmentInd];
      segmentInd += shiftInd;
      Int_t s2 = pols[segmentInd];
      segmentInd += shiftInd;
      Int_t segEnds[] = {segs[s1 * 3 + 1], segs[s1 * 3 + 2],
                         segs[s2 * 3 + 1], segs[s2 * 3 + 2]};
      Int_t numPnts[3] = {0};

      if (segEnds[0] == segEnds[2]) {
         numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[3];
      } else if (segEnds[0] == segEnds[3]) {
         numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[2];
      } else if (segEnds[1] == segEnds[2]) {
         numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[3];
      } else {
         numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[2];
      }

      fPolyDesc[currInd] = 3;
      Int_t sizeInd = currInd++;
      fPolyDesc[currInd++] = numPnts[0];
      fPolyDesc[currInd++] = numPnts[1];
      fPolyDesc[currInd++] = numPnts[2];
      Int_t lastAdded = numPnts[2];

      Int_t end = shiftInd < 0 ? j + 1 : j + segmentCol;
      for (; segmentInd != end; segmentInd += shiftInd) {
         segEnds[0] = segs[pols[segmentInd] * 3 + 1];
         segEnds[1] = segs[pols[segmentInd] * 3 + 2];
         if (segEnds[0] == lastAdded) {
            fPolyDesc[currInd++] = segEnds[1];
            lastAdded = segEnds[1];
         } else {
            fPolyDesc[currInd++] = segEnds[0];
            lastAdded = segEnds[0];
         }
         ++fPolyDesc[sizeInd];
      }
      j += segmentCol + 2;
   }
   }
   CalculateNormals();

}

//______________________________________________________________________________
void TGLFaceSet::GLDraw(const TGLFrustum *fr)const
{
   if (fr) {
      if (!fr->ClipOnBoundingBox(*this)) return;
   }

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   glLoadName(GetGLName());
   GLDrawPolys();

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }

   if (fIsSelected) {
      fSelectionBox.DrawBox();
   }
}

//______________________________________________________________________________
void TGLFaceSet::GLDrawPolys()const
{
  GLUtriangulatorObj *tessObj = GetTesselator();
  const Double_t *pnts = &fVertices[0];
  const Double_t *normals = &fNormals[0];
  const Int_t *pols = &fPolyDesc[0];

   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t npoints = pols[j++];

      if (tessObj && npoints > 4) {
         gluBeginPolygon(tessObj);
         gluNextContour(tessObj, (GLenum)GLU_UNKNOWN);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j) {
            gluTessVertex(tessObj, (Double_t *)pnts + pols[j] * 3, (Double_t *)pnts + pols[j] * 3);
         }
         gluEndPolygon(tessObj);
      } else {
         glBegin(GL_POLYGON);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j) {
            glVertex3dv(pnts + pols[j] * 3);
         }
         glEnd();
      }
   }
}

//______________________________________________________________________________
Int_t TGLFaceSet::CheckPoints(const Int_t *source, Int_t *dest) const
{
   const Double_t * p1 = &fVertices[source[0] * 3];
   const Double_t * p2 = &fVertices[source[1] * 3];
   const Double_t * p3 = &fVertices[source[2] * 3];
   Int_t retVal = 1;

   if (Eq(p1, p2)) {
      dest[0] = source[0];
      if (!Eq(p1, p3) ) {
         dest[1] = source[2];
         retVal = 2;
      }
   } else if (Eq(p1, p3)) {
      dest[0] = source[0];
      dest[1] = source[1];
      retVal = 2;
   } else {
      dest[0] = source[0];
      dest[1] = source[1];
      retVal = 2;
      if (!Eq(p2, p3)) {
         dest[2] = source[2];
         retVal = 3;
      }
   }

   return retVal;
}

//______________________________________________________________________________
Bool_t TGLFaceSet::Eq(const Double_t *p1, const Double_t *p2)
{
   Double_t dx = TMath::Abs(p1[0] - p2[0]);
   Double_t dy = TMath::Abs(p1[1] - p2[1]);
   Double_t dz = TMath::Abs(p1[2] - p2[2]);
   return dx < 1e-10 && dy < 1e-10 && dz < 1e-10;
}

//______________________________________________________________________________
void TGLFaceSet::CalculateNormals()
{
   Double_t *pnts = &fVertices[0];
   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t polEnd = fPolyDesc[j] + j + 1;
      Int_t norm[] = {fPolyDesc[j + 1], fPolyDesc[j + 2], fPolyDesc[j + 3]};
      j += 4;
      Int_t check = CheckPoints(norm, norm), ngood = check;
      if (check == 3) {
         TMath::Normal2Plane(pnts + norm[0] * 3, pnts + norm[1] * 3,
                             pnts + norm[2] * 3, &fNormals[i * 3]);
         j = polEnd;
         continue;
      }
      while (j < (UInt_t)polEnd) {
         norm[ngood++] = fPolyDesc[j++];
         if (ngood == 3) {
            ngood = CheckPoints(norm, norm);
            if (ngood == 3) {
               TMath::Normal2Plane(pnts + norm[0] * 3, pnts + norm[1] * 3,
                                   pnts + norm[2] * 3, &fNormals[i * 3]);
               j = polEnd;
               break;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGLFaceSet::Stretch(Double_t xs, Double_t ys, Double_t zs)
{
   TGLSceneObject::Stretch(xs, ys, zs);
   CalculateNormals();
}

//______________________________________________________________________________
TGLPolyMarker::TGLPolyMarker(const TBuffer3D &buffer, const Float_t *c, UInt_t n, TObject *r)
                  :TGLSceneObject(buffer, c, n, r),
                   fStyle(7)
{
   //TAttMarker is not TObject descendant, so I need dynamic_cast
   if (TAttMarker *realObj = dynamic_cast<TAttMarker *>(buffer.fID))
      fStyle = realObj->GetMarkerStyle();
}

//______________________________________________________________________________
void TGLPolyMarker::GLDraw(const TGLFrustum *fr)const
{
   if (fr) {
      if (!fr->ClipOnBoundingBox(*this)) return;
   }

   const Double_t *vertices = &fVertices[0];
   UInt_t size = fVertices.size();
   Int_t stacks = 6, slices = 6;
   Float_t pointSize = 6.f;
   Double_t topRadius = 5.;
   GLUquadric *quadObj = GetQuadric();

   glLoadName(GetGLName());
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);

   switch (fStyle) {
   case 27:
      stacks = 2, slices = 4;
   case 4:case 8:case 20:case 24:
      if (quadObj) {
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluSphere(quadObj, 5., slices, stacks);
            glPopMatrix();
         }
      }
      break;
   case 22:case 26:
      topRadius = 0.;
   case 21:case 25:
      if (quadObj) {
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluCylinder(quadObj, 5., topRadius, 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 23:
      if (quadObj) {
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            glRotated(180, 1., 0., 0.);
            gluCylinder(quadObj, 5., 0., 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 3: case 2: case 5:
      DrawStars();
      break;
   case 1: case 9: case 10: case 11: default:{
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
   }
   break;
   case 6:
      pointSize = 3.f;
   case 7:
      glPointSize(pointSize);
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
      glPointSize(1.f);
   }

   if (fIsSelected) {
      fSelectionBox.DrawBox();
   }
}

//______________________________________________________________________________
void TGLPolyMarker::DrawStars()const
{
   glDisable(GL_LIGHTING);
   glColor3fv(fColor);
   for (UInt_t i = 0; i < fVertices.size(); i += 3) {
      Double_t x = fVertices[i];
      Double_t y = fVertices[i + 1];
      Double_t z = fVertices[i + 2];
      glBegin(GL_LINES);
      if (fStyle == 2 || fStyle == 3) {
         glVertex3d(x - 2., y, z);
         glVertex3d(x + 2., y, z);
         glVertex3d(x, y, z - 2.);
         glVertex3d(x, y, z + 2.);
         glVertex3d(x, y - 2., z);
         glVertex3d(x, y + 2., z);
      }
      if(fStyle != 2) {
         glVertex3d(x - 1.4, y - 1.4, z - 1.4);
         glVertex3d(x + 1.4, y + 1.4, z + 1.4);
         glVertex3d(x - 1.4, y - 1.4, z + 1.4);
         glVertex3d(x + 1.4, y + 1.4, z - 1.4);
         glVertex3d(x - 1.4, y + 1.4, z - 1.4);
         glVertex3d(x + 1.4, y - 1.4, z + 1.4);
         glVertex3d(x - 1.4, y + 1.4, z + 1.4);
         glVertex3d(x + 1.4, y - 1.4, z - 1.4);
      }
      glEnd();
   }
   glEnable(GL_LIGHTING);
}

//______________________________________________________________________________
TGLPolyLine::TGLPolyLine(const TBuffer3D &buffer, const Float_t *c, UInt_t n, TObject *r)
                :TGLSceneObject(buffer, c, n, r)
{
}

//______________________________________________________________________________
void TGLPolyLine::GLDraw(const TGLFrustum *fr)const
{
   if (fr) {
      if (!fr->ClipOnBoundingBox(*this)) return;
   }

   glLoadName(GetGLName());
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glBegin(GL_LINE_STRIP);

   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);

   glEnd();

   if (fIsSelected) {
      fSelectionBox.DrawBox();
   }
}

UInt_t TGLSphere::fSphereList = 0;

//______________________________________________________________________________
TGLSphere::TGLSphere(const TBuffer3DSphere &buffer, const Float_t *c, UInt_t n, TObject *r)
                :TGLSceneObject(buffer, c, n, r)
{
   // Default ctor
   // TODO: Can get this from BB at present as we know it is full
   // sphere. When cut we need to extract translation from local master matrix
   fX      = (buffer.fBBLowVertex[0] + buffer.fBBLowVertex[0])/2.0;
   fY      = (buffer.fBBLowVertex[1] + buffer.fBBLowVertex[1])/2.0;
   fZ      = (buffer.fBBLowVertex[2] + buffer.fBBLowVertex[2])/2.0;
   fRadius = buffer.fRadiusOuter;

   // TODO: 
   // Support hollow & cut spheres
   // buffer.fRadiusInner;
   // buffer.fThetaMin;
   // buffer.fThetaMax;
   // buffer.fPhiMin;
   // buffer.fPhiMax;

   fNdiv   = 20; // Same hardcoded value as passed through buffer previously
                 // This will come from viewer LOD scheme on draw in future
}

//______________________________________________________________________________
void TGLSphere::BuildList()
{
   if (!(fSphereList = glGenLists(1))) {
      ::Error("TGLSphere::BuildList", "Could not build display list for sphere\n");
      return;
   }

   if (GLUquadric *quadObj = GetQuadric()) {
      glNewList(fSphereList, GL_COMPILE);
      gluSphere(quadObj, 1, 20, 20);
      glEndList();
   } else {
      fSphereList = 0;
   }
}


//______________________________________________________________________________
void TGLSphere::GLDraw(const TGLFrustum *fr)const
{
   if (!fSphereList) {
      BuildList();
      if(!fSphereList) return;
   }

   if (fr) {
      if (!fr->ClipOnBoundingBox(*this)) return;
   }

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   glEnable(GL_NORMALIZE);

   glLoadName(GetGLName());
   glPushMatrix();
   glTranslated(fX, fY, fZ);
   if (fRadius > 1.) glScaled(fRadius, fRadius, fRadius);
   glCallList(fSphereList);
   glPopMatrix();

   glDisable(GL_NORMALIZE);

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }

   if (fIsSelected) {
      fSelectionBox.DrawBox();
   }
}

//______________________________________________________________________________
void TGLSphere::Shift(Double_t x, Double_t y, Double_t z)
{
   fX += x;
   fY += y;
   fZ += z;
   fSelectionBox.Shift(x, y, z);
}

//______________________________________________________________________________
void TGLSphere::Stretch(Double_t, Double_t, Double_t)
{
}

////////////////////////////////////////////////////////////
namespace GL{
   struct Vertex3d {
      Double_t fXYZ[3];
      Double_t &operator [] (Int_t ind) {return fXYZ[ind];}
      Double_t operator [] (Int_t ind)const{return fXYZ[ind];}

      void Negate()
      {
         fXYZ[0] = -fXYZ[0];
         fXYZ[1] = -fXYZ[1];
         fXYZ[2] = -fXYZ[2];
      }
   };
}

using GL::Vertex3d;

Vertex3d lowNormal = {{0., 0., -1.}};
Vertex3d highNormal = {{0., 0., 1.}};

class TGLMesh {
protected:
   Double_t fRmin1, fRmax1, fRmin2, fRmax2;
   Double_t fDz;
   Vertex3d fCenter;
   //normals for top and bottom (for cuts)
   Vertex3d fNlow;
   Vertex3d fNhigh;

   enum {kLod = 40};

   void GetNormal(const Vertex3d &vertex, Vertex3d &normal)const;
   Double_t GetZcoord(Double_t x, Double_t y, Double_t z)const;
   const Vertex3d &MakeVertex(Double_t x, Double_t y, Double_t z)const;
   
public:
   TGLMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz, 
                   const Vertex3d &center, const Vertex3d &l = lowNormal, 
                   const Vertex3d &h = highNormal);

   void Shift(Double_t xs, Double_t ys, Double_t zs)
   {
      fCenter[0] += xs; fCenter[1] += ys; fCenter[2] += zs;
   }
      
   virtual void Draw(const Double_t *rot)const = 0;
};

//segment contains 3 quad strips:
//one for inner and outer sides, two for top and bottom
class TubeSegMesh : public TGLMesh {
private:
   Vertex3d fMesh[(kLod + 1) * 8 + 8];
   Vertex3d fNorm[(kLod + 1) * 8 + 8];

public:
   TubeSegMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
               Double_t phi1, Double_t phi2, const Vertex3d &center, 
               const Vertex3d &l = lowNormal, const Vertex3d &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//four quad strips:
//outer, inner, top, bottom
class TubeMesh : public TGLMesh {
private:
   Vertex3d fMesh[(kLod + 1) * 8];
   Vertex3d fNorm[(kLod + 1) * 8];

public:
   TubeMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
            const Vertex3d &center, const Vertex3d &l = lowNormal, 
            const Vertex3d &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle funs
class CylinderMesh : public TGLMesh {
private:
   Vertex3d fMesh[(kLod + 1) * 4 + 2];
   Vertex3d fNorm[(kLod + 1) * 4 + 2];

public:
   CylinderMesh(Double_t r1, Double_t r2, Double_t dz, const Vertex3d &center,
                const Vertex3d &l = lowNormal, const Vertex3d &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle fans
class CylinderSegMesh : public TGLMesh {
private:
   Vertex3d fMesh[(kLod + 1) * 4 + 10];
   Vertex3d fNorm[(kLod + 1) * 4 + 10];

public:
   CylinderSegMesh(Double_t r1, Double_t r2, Double_t dz, Double_t phi1, Double_t phi2,
                   const Vertex3d &center, const Vertex3d &l = lowNormal, 
                   const Vertex3d &h = highNormal);

   void Draw(const Double_t *rot)const;
};


//______________________________________________________________________________
TGLMesh::TGLMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz, 
                               const Vertex3d &c, const Vertex3d &l, const Vertex3d &h)
                     :fRmin1(r1), fRmax1(r2), fRmin2(r3), fRmax2(r4),
                      fDz(dz), fCenter(c), fNlow(l), fNhigh(h)
{
}

//______________________________________________________________________________
void TGLMesh::GetNormal(const Vertex3d &v, Vertex3d &n)const
{
   Double_t z = (fRmax1 - fRmax2) / (2 * fDz);
   Double_t mag = TMath::Sqrt(v[0] * v[0] + v[1] * v[1] + z * z);

   n[0] = v[0] / mag;
   n[1] = v[1] / mag;
   n[2] = z / mag;
}

//______________________________________________________________________________
Double_t TGLMesh::GetZcoord(Double_t x, Double_t y, Double_t z)const
{
   Double_t newz = 0;
   if (z < 0) newz = -fDz - (x * fNlow[0] + y * fNlow[1]) / fNlow[2];
   else newz = fDz - (x * fNhigh[0] + y * fNhigh[1]) / fNhigh[2];

   return newz;
}

//______________________________________________________________________________
const Vertex3d &TGLMesh::MakeVertex(Double_t x, Double_t y, Double_t z)const
{
   static Vertex3d vert = {{0., 0., 0.}};
   vert[0] = x;
   vert[1] = y;
   vert[2] = GetZcoord(x, y, z);

   return vert;
}

//______________________________________________________________________________
TubeSegMesh::TubeSegMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz, 
                         Double_t phi1, Double_t phi2, const Vertex3d &center, 
                         const Vertex3d &l, const Vertex3d &h)
                 :TGLMesh(r1, r2, r3, r4, dz, center, l, h), fMesh(), fNorm()
                      
{
   const Double_t delta = (phi2 - phi1) / kLod;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);
   const Int_t topShift = (kLod + 1) * 4 + 8;
   const Int_t botShift = (kLod + 1) * 6 + 8;
   Int_t j = 4 * (kLod + 1) + 2;

   //defining all three strips here, first strip is non-closed here
   for (Int_t i = 0, e = (kLod + 1) * 2; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         GetNormal(fMesh[j], fNorm[j]);
         fNorm[j].Negate();
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         fMesh[j + 1] = MakeVertex(fRmin1 * c, fRmin1 * s, -fDz);
         fMesh[i + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmin1 * c, fRmin1 * s, - fDz);
         GetNormal(fMesh[j + 1], fNorm[j + 1]);
         fNorm[j + 1].Negate();
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         j -= 2;
      }
   
      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }

   //closing first strip
   Int_t ind = 2 * (kLod + 1);
   Vertex3d norm = {{0., 0., 0.}};

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[ind + 4];
   fMesh[ind + 3] = fMesh[ind + 5];
   TMath::Normal2Plane(fMesh[ind].fXYZ, fMesh[ind + 1].fXYZ, fMesh[ind + 2].fXYZ,
                       norm.fXYZ);
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind = topShift - 4;
   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].fXYZ, fMesh[ind + 1].fXYZ, fMesh[ind + 2].fXYZ,
                       norm.fXYZ);
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

//______________________________________________________________________________
void TubeSegMesh::Draw(const Double_t *rot)const
{
   glPushMatrix();
   glTranslated(fCenter[0], fCenter[1], fCenter[2]);
   glMultMatrixd(rot);

   //Tube segment is drawn as three quad strips
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d), fNorm[0].fXYZ);
   //3. draw first strip
   glDrawArrays(GL_QUAD_STRIP, 0, 4 * (kLod + 1) + 8);
   //4. draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (kLod + 1) + 8, 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (kLod + 1) + 8, 2 * (kLod + 1));

   glDisableClientState(GL_VERTEX_ARRAY);   
   glDisableClientState(GL_NORMAL_ARRAY);

   glPopMatrix();
}

//______________________________________________________________________________
TubeMesh::TubeMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t z,
                      const Vertex3d &center, const Vertex3d &l, const Vertex3d &h)
             :TGLMesh(r1, r2, r3, r4, z, center, l, h), fMesh(), fNorm()
{
   const Double_t delta = TMath::TwoPi() / kLod;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const Int_t topShift = (kLod + 1) * 4;
   const Int_t botShift = (kLod + 1) * 6;
   Int_t j = 4 * (kLod + 1) - 2;

   //defining all four strips here
   for (Int_t i = 0, e = (kLod + 1) * 2; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         GetNormal(fMesh[j], fNorm[j]);
         fNorm[j].Negate();
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         fMesh[j + 1] = MakeVertex(fRmin1 * c, fRmin1 * s, -fDz);
         fMesh[i + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmin1 * c, fRmin1 * s, - fDz);
         GetNormal(fMesh[j + 1], fNorm[j + 1]);
         fNorm[j + 1].Negate();
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         j -= 2;
      }
      
      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }
}

//______________________________________________________________________________
void TubeMesh::Draw(const Double_t *rot)const
{
   glPushMatrix();
   glTranslated(fCenter[0], fCenter[1], fCenter[2]);
   glMultMatrixd(rot);

   //Tube is drawn as four quad strips
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d), fNorm[0].fXYZ);
   //draw outer and inner strips
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 2 * (kLod + 1), 2 * (kLod + 1));
   //draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (kLod + 1), 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (kLod + 1), 2 * (kLod + 1));
   //5. disabling vertex arrays   
   glDisableClientState(GL_VERTEX_ARRAY);   
   glDisableClientState(GL_NORMAL_ARRAY);

   glPopMatrix();
}

//______________________________________________________________________________
CylinderMesh::CylinderMesh(Double_t r1, Double_t r2, Double_t dz, const Vertex3d &center, 
                           const Vertex3d &l, const Vertex3d &h)
                 :TGLMesh(0., r1, 0., r2, dz, center, l, h), fMesh(), fNorm()
{
   const Double_t delta = TMath::TwoPi() / kLod;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   //central point of top fan
   Int_t topShift = (kLod + 1) * 2;
   fMesh[topShift][0] = fMesh[topShift][1] = 0., fMesh[topShift][2] = fDz;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //central point of bottom fun
   Int_t botShift = topShift + 2 * (kLod + 1);
   fMesh[botShift][0] = fMesh[botShift][1] = 0., fMesh[botShift][2] = -fDz;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining 1 strip and 2 fans
   for (Int_t i = 0, e = (kLod + 1) * 2, j = 0; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[j + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         ++j;
      }
   
      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }
}

//______________________________________________________________________________
void CylinderMesh::Draw(const Double_t *rot)const
{
   glPushMatrix();
   glTranslated(fCenter[0], fCenter[1], fCenter[2]);
   glMultMatrixd(rot);

   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d), fNorm[0].fXYZ);

   //draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1));
   //draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (kLod + 1), kLod + 2);
   glDrawArrays(GL_TRIANGLE_FAN, 3 * (kLod + 1) + 1, kLod + 2);

   glDisableClientState(GL_VERTEX_ARRAY);   
   glDisableClientState(GL_NORMAL_ARRAY);

   glPopMatrix();
}

//______________________________________________________________________________
   CylinderSegMesh::CylinderSegMesh(Double_t r1, Double_t r2, Double_t dz, Double_t phi1,
                                    Double_t phi2, const Vertex3d &center, const Vertex3d &l, 
                                    const Vertex3d &h)
                     :TGLMesh(0., r1, 0., r2, dz, center, l, h), fMesh(), fNorm()
{
   //One quad mesh and two fans
   Double_t delta = (phi2 - phi1) / kLod;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const Vertex3d vTop = {{0., 0., fDz}};
   const Vertex3d vBot = {{0., 0., - fDz}};

   //center of top fan
   Int_t topShift = (kLod + 1) * 2 + 8;
   fMesh[topShift] = vTop;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //center of bottom fan
   Int_t botShift = topShift + kLod + 1;
   fMesh[botShift] = vBot;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining strip and two fans
   //strip is not closed here
   Int_t i = 0;
   for (Int_t e = (kLod + 1) * 2, j = 0; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kFALSE;
         fNorm[j + topShift] = fNhigh;
         fNorm[j + botShift] = fNlow;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         ++j;
      }
   
      GetNormal(fMesh[i], fNorm[i]);
   }

   //closing first strip
   Int_t ind = 2 * (kLod + 1);
   Vertex3d norm = {{0., 0., 0.}};

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = vTop;
   fMesh[ind + 3] = vBot;
   TMath::Normal2Plane(fMesh[ind].fXYZ, fMesh[ind + 1].fXYZ, fMesh[ind + 2].fXYZ,
                          norm.fXYZ);
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind += 4;
   fMesh[ind] = vTop;
   fMesh[ind + 1] = vBot;
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].fXYZ, fMesh[ind + 1].fXYZ, fMesh[ind + 2].fXYZ,
                       norm.fXYZ);
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

//______________________________________________________________________________
void CylinderSegMesh::Draw(const Double_t *rot)const
{
   glPushMatrix();
   glTranslated(fCenter[0], fCenter[1], fCenter[2]);
   glMultMatrixd(rot);

   //Cylinder segment is drawn as one quad strip and
   //two triangle fans
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d), fNorm[0].fXYZ);
   //3. draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1) + 8);
   //4. draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (kLod + 1) + 8, kLod + 2);
   //      glDrawArrays(GL_TRIANGLE_FAN, 3 * (kLod + 1) + 9, kLod + 2);
   //5. disabling vertex arrays   
   glDisableClientState(GL_VERTEX_ARRAY);   
   glDisableClientState(GL_NORMAL_ARRAY);

   glPopMatrix();
}

//______________________________________________________________________________
TGLCylinder::TGLCylinder(const TBuffer3DTube &buffer, const Float_t *c, UInt_t n, TObject *r)
            :TGLSceneObject(buffer, 16, c, n, r)
{
   fInv = buffer.fReflection;
   CreateParts(buffer);
}

//______________________________________________________________________________
TGLCylinder::~TGLCylinder()
{
   for (UInt_t i = 0; i < fParts.size(); ++i) {
      delete fParts[i];
      fParts[i] = 0;//not to have invalid pointer for pseudo-destructor call :)
   }
}

//______________________________________________________________________________
void TGLCylinder::CreateParts(const TBuffer3DTube &buffer)
{
   Double_t r1 = buffer.fRadiusInner;
   Double_t r2 = buffer.fRadiusOuter;
   Double_t r3 = buffer.fRadiusInner;
   Double_t r4 = buffer.fRadiusOuter;
   Double_t dz = buffer.fHalfLength;

   // Stuff the transposed rotation component of the local -> master 
   // translation matrix into verticies array
   // Then stuff the translation component in to 'center' - as before
   // TODO: Clean this up - will be tidied as part of local frame conversion
   // of whole viewer - don't forget to remove the hack on TGeo side
   // Shapes which can't provide local frame + trans matrix will not be able
   // to use these tube drawing routines -> raw tesselation
   const Double_t * lm = buffer.fLocalMaster;

   // Note buffer contains row major matrix currently
   // TODO: Decide on column or row major and comment in TBuffer3D
   fVertices[0] =  lm[0]; fVertices[1] =  lm[4]; fVertices[2] =  lm[8];  fVertices[3] =  0.0;
   fVertices[4] =  lm[1]; fVertices[5] =  lm[5]; fVertices[6] =  lm[9];  fVertices[7] =  0.0;
   fVertices[8] =  lm[2]; fVertices[9] =  lm[6]; fVertices[10] = lm[10]; fVertices[11] = 0.0;
   fVertices[12] = 0.0;   fVertices[13] = 0.0;   fVertices[14] = 0.0;    fVertices[15] = 1.0;

   Vertex3d center = {{lm[12], lm[13], lm[14]}};

   switch (buffer.Type()) {
   case TBuffer3DTypes::kTube:
      {
         Vertex3d low = {{0., 0., -1.}};
         Vertex3d high = {{0., 0., 1.}};
         fParts.push_back(new TubeMesh(r1, r2, r3, r4, dz, center, low, high));
      }
      break;
   case TBuffer3DTypes::kTubeSeg:
      {
         const TBuffer3DTubeSeg * segBuffer = dynamic_cast<const TBuffer3DTubeSeg *>(&buffer);
         if (!segBuffer) { 
            assert(kFALSE); 
            return; 
         }

         Double_t phi1 = segBuffer->fPhiMin;
         Double_t phi2 = segBuffer->fPhiMax;
			if (phi2 < phi1) phi2 += 360.;
			phi1 *= TMath::DegToRad();
			phi2 *= TMath::DegToRad();
        
         // TODO: Check with Timur what this means - was hardcoded into the buffer
         // on TGeo side - so hardcoded here now....same as above for kTUBE
         Vertex3d low = {{0., 0., -1.}}; // Vertex3d low = {{p[44], p[45], p[46]}};
         Vertex3d high = {{0., 0., 1.}}; // Vertex3d high = {{p[47], p[48], p[49]}};    
         fParts.push_back(new TubeSegMesh(r1, r2, r3, r4, dz, phi1, 
                                          phi2, center, low, high));
      }
      break;
   default:;
   //polycone should be here
   }  
}

//______________________________________________________________________________
void TGLCylinder::GLDraw(const TGLFrustum *fr)const
{
   if (fr) {
      if (!fr->ClipOnBoundingBox(*this)) return;
   }

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);
   glLoadName(GetGLName());

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   if (fInv) glFrontFace(GL_CW);
   //draw here
   for (UInt_t i = 0; i < fParts.size(); ++i) fParts[i]->Draw(&fVertices[0]);
   if (fInv) glFrontFace(GL_CCW);

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }

   if (fIsSelected) {
      fSelectionBox.DrawBox();
   }
}

//______________________________________________________________________________
void TGLCylinder::Shift(Double_t xs, Double_t ys, Double_t zs)
{
   fSelectionBox.Shift(xs, ys, zs);
   for (UInt_t i = 0; i < fParts.size(); ++i)
      fParts[i]->Shift(xs, ys, zs);
}

//______________________________________________________________________________
void TGLCylinder::Stretch(Double_t, Double_t, Double_t)
{
   //non-stretchable now
}

