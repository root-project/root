// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.cxx,v 1.20 2004/11/26 11:08:05 brun Exp $
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
#include "TError.h"

#include "TGLSceneObject.h"

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

static GLenum gLightNames[] = {GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3,
                               GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7};
//______________________________________________________________________________
TGLSelection::TGLSelection()
{
}

//______________________________________________________________________________
TGLSelection::TGLSelection(const PDD_t &x, const PDD_t &y, const PDD_t &z)
                 :fRangeX(x), fRangeY(y), fRangeZ(z)
{
}

//______________________________________________________________________________
void TGLSelection::DrawBox()const
{
   Double_t xmin = fRangeX.first, xmax = fRangeX.second;
   Double_t ymin = fRangeY.first, ymax = fRangeY.second;
   Double_t zmin = fRangeZ.first, zmax = fRangeZ.second;
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);

   glColor3f(1.f, 1.f, 1.f);
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
void TGLSelection::SetBox(const PDD_t &x, const PDD_t &y, const PDD_t &z)
{
   fRangeX.first = x.first, fRangeX.second = x.second;
   fRangeY.first = y.first, fRangeY.second = y.second;
   fRangeZ.first = z.first, fRangeZ.second = z.second;
}

//______________________________________________________________________________
void TGLSelection::Shift(Double_t x, Double_t y, Double_t z)
{
   fRangeX.first += x, fRangeX.second += x;
   fRangeY.first += y, fRangeY.second += y;
   fRangeZ.first += z, fRangeZ.second += z;
}

//______________________________________________________________________________
void TGLSelection::Stretch(Double_t xs, Double_t ys, Double_t zs)
{
   fRangeX.first *= xs, fRangeX.second *= xs;
   fRangeY.first *= ys, fRangeY.second *= ys;
   fRangeZ.first *= zs, fRangeZ.second *= zs;
}

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const Double_t *start, const Double_t *end,
                               const Float_t *color, UInt_t glName, TObject *obj)
                   :fVertices(start, end), fColor(),
                    fGLName(glName), fNextT(0), fRealObject(obj)
{
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
   SetBox();
}

//______________________________________________________________________________
Bool_t TGLSceneObject::IsTransparent()const
{
   return kFALSE;
}

//______________________________________________________________________________
void TGLSceneObject::ResetTransparency(char)
{
}

//______________________________________________________________________________
void TGLSceneObject::Shift(Double_t, Double_t, Double_t)
{
}

//______________________________________________________________________________
void TGLSceneObject::Stretch(Double_t, Double_t, Double_t)
{
}

//______________________________________________________________________________
void TGLSceneObject::SetColor(const Float_t *newColor)
{
   for (Int_t i = 0; i < 17; ++i) fColor[i] = newColor[i];
}

//______________________________________________________________________________
void TGLSceneObject::SetBox()
{
   typedef std::pair<Double_t, Double_t>PDD_t;
   PDD_t xb(fVertices[0], fVertices[0]);
   PDD_t yb(fVertices[1], fVertices[1]);
   PDD_t zb(fVertices[2], fVertices[2]);
   for (Int_t nv = 3, e = fVertices.size(); nv < e; nv += 3) {
      xb.first = TMath::Min(xb.first, fVertices[nv]);
      xb.second = TMath::Max(xb.second, fVertices[nv]);
      yb.first = TMath::Min(yb.first, fVertices[nv + 1]);
      yb.second = TMath::Max(yb.second, fVertices[nv + 1]);
      zb.first = TMath::Min(zb.first, fVertices[nv + 2]);
      zb.second = TMath::Max(zb.second, fVertices[nv + 2]);
   }
   fSelectionBox.SetBox(xb, yb, zb);
   fCenter[0] = xb.first + (xb.second - xb.first) / 2;
   fCenter[1] = yb.first + (yb.second - yb.first) / 2;
   fCenter[2] = zb.first + (zb.second - zb.first) / 2;
}

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buff, const Float_t *color,
                       UInt_t glname, TObject *realobj)
               :TGLSceneObject(buff.fPnts, buff.fPnts + 3 * buff.fNbPnts, color, glname, realobj),
                fNormals(3 * buff.fNbPols)
{
   fColor[3] = 1.f - buff.fTransparency / 100.f;
   fNbPols = buff.fNbPols;

   Int_t *segs = buff.fSegs;
   Int_t *pols = buff.fPols;
   Int_t shiftInd = buff.TestBit(TBuffer3D::kIsReflection) ? 1 : -1;

   for (Int_t numPol = 0, j = 1; numPol < buff.fNbPols; ++numPol) {
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

      fPolyDesc.push_back(3);
      Int_t sizeInd = fPolyDesc.size() - 1;
      fPolyDesc.insert(fPolyDesc.end(), numPnts, numPnts + 3);
      Int_t lastAdded = numPnts[2];

      Int_t end = shiftInd < 0 ? j + 1 : j + segmentCol + 1;
      for (; segmentInd != end; segmentInd += shiftInd) {
         segEnds[0] = segs[pols[segmentInd] * 3 + 1];
         segEnds[1] = segs[pols[segmentInd] * 3 + 2];
         if (segEnds[0] == lastAdded) {
            fPolyDesc.push_back(segEnds[1]);
            lastAdded = segEnds[1];
         } else {
            fPolyDesc.push_back(segEnds[0]);
            lastAdded = segEnds[0];
         }
         ++fPolyDesc[sizeInd];
      }
      j += segmentCol + 2;
   }
   CalculateNormals();
}

//______________________________________________________________________________
Bool_t TGLFaceSet::IsTransparent()const
{
   return fColor[3] < 1.f;;
}

//______________________________________________________________________________
void TGLFaceSet::ResetTransparency(char newval)
{
   fColor[3] = 1.f - newval / 100.f;
}

//______________________________________________________________________________
void TGLFaceSet::GLDraw()const
{
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, fColor + 4);
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor + 8);
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor + 12);
   glMaterialf(GL_FRONT, GL_SHININESS, fColor[16]);

   GLUtriangulatorObj *tessObj = GetTesselator();
   const Double_t *pnts = &fVertices[0];
   const Double_t *normals = &fNormals[0];
   const Int_t *pols = &fPolyDesc[0];

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   glLoadName(GetGLName());

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

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }
}

//______________________________________________________________________________
void TGLFaceSet::Shift(Double_t x, Double_t y, Double_t z)
{
   for (UInt_t i = 0, e = fVertices.size(); i < e; i += 3) {
      fVertices[i] += x;
      fVertices[i + 1] += y;
      fVertices[i + 2] += z;
   }
   SetBox();
}

//______________________________________________________________________________
void TGLFaceSet::Stretch(Double_t xs, Double_t ys, Double_t zs)
{
   fSelectionBox.Shift(-fCenter[0], -fCenter[1], -fCenter[2]);
   fSelectionBox.Stretch(xs, ys, zs);
   fSelectionBox.Shift(fCenter[0], fCenter[1], fCenter[2]);
   Shift(-fCenter[0], -fCenter[1], -fCenter[2]);
   for (UInt_t i = 0, e = fVertices.size(); i < e; i += 3) {
      fVertices[i] *= xs;
      fVertices[i + 1] *= ys;
      fVertices[i + 2] *= zs;
   }
   Shift(fCenter[0], fCenter[1], fCenter[2]);
   CalculateNormals();
}

//______________________________________________________________________________
Int_t TGLFaceSet::CheckPoints(const Int_t * source, Int_t *dest) const
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
TGLPolyMarker::TGLPolyMarker(const TBuffer3D &b, const Float_t *c, UInt_t n, TObject *r)
                  :TGLSceneObject(b.fPnts, b.fPnts + 3 * b.fNbPnts, c, n, r),
                   fStyle(7)
{
   //TAttMarker is not TObject descendant, so I need dynamic_cast
   if (TAttMarker *realObj = dynamic_cast<TAttMarker *>(b.fId))
      fStyle = realObj->GetMarkerStyle();
}

//______________________________________________________________________________
void TGLPolyMarker::GLDraw()const
{
   const Double_t *vertices = &fVertices[0];
   UInt_t size = fVertices.size();
   Int_t stacks = 6, slices = 6;
   Float_t pointSize = 6.f;
   Double_t top_radius = 5.;
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
      top_radius = 0.;
   case 21:case 25:
      if (quadObj) {
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluCylinder(quadObj, 5., top_radius, 5., 4, 1);
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
TGLPolyLine::TGLPolyLine(const TBuffer3D &b, const Float_t *c, UInt_t n, TObject *r)
                :TGLSceneObject(b.fPnts, b.fPnts + 3 * b.fNbPnts, c, n, r)
{
}

//______________________________________________________________________________
void TGLPolyLine::GLDraw()const
{
   glLoadName(GetGLName());
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glBegin(GL_LINE_STRIP);

   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);

   glEnd();
}

//______________________________________________________________________________
TGLSphere::TGLSphere(const TBuffer3D &b, const Float_t *c, UInt_t n, TObject *r)
                :TGLSceneObject(b.fPnts, b.fPnts + 3 * b.fNbPnts, c, n, r)
{
   // Default ctor
   fX      = b.fPnts[0];
   fY      = b.fPnts[1];
   fZ      = b.fPnts[2];
   fNdiv   = (Int_t)b.fPnts[9];
   fRadius = b.fPnts[10];
}

//______________________________________________________________________________
Bool_t TGLSphere::IsTransparent()const
{
   return fColor[3] < 1.f;;
}

//______________________________________________________________________________
void TGLSphere::GLDraw()const
{
   // Draw a Sphere using OpenGL Sphere primitive gluSphere
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

   if (GLUquadric *quadObj = GetQuadric()) {
      glLoadName(GetGLName());
      glPushMatrix();
      glTranslated(fX, fY, fZ);
      gluSphere(quadObj, fRadius, fNdiv, fNdiv);
      glPopMatrix();
   }

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }
}

//______________________________________________________________________________
void TGLSphere::Shift(Double_t x, Double_t y, Double_t z)
{
   fX += x;
   fY += y;
   fZ += z;
}

//______________________________________________________________________________
TGLTube::TGLTube(const TBuffer3D &b, const Float_t *c, UInt_t n, TObject *r)
            :TGLSceneObject(b.fPnts, b.fPnts + 3 * b.fNbPnts, c, n, r)
{
   fColor[3] = 1.f - b.fTransparency / 100.f;

   fX = b.fPnts[0];
   fY = b.fPnts[1];
   fZ = b.fPnts[2];
   fNdiv = (Int_t)b.fPnts[9];

   fRmin1 = b.fPnts[10];
   fRmax1 = b.fPnts[11];
   fRmin2 = b.fPnts[12];
   fRmax2 = b.fPnts[13];
   fDz = b.fPnts[14];

   const Double_t *p = b.fPnts;

   fRotM[0] = p[15], fRotM[1] = p[18], fRotM[2] = p[21], fRotM[3] = 0.;
   fRotM[4] = p[16], fRotM[5] = p[19], fRotM[6] = p[22], fRotM[7] = 0.;
   fRotM[8] = p[17], fRotM[9] = p[20], fRotM[10] = p[23], fRotM[11] = 0.;
   fRotM[12] = 0.,    fRotM[13] = 0.,    fRotM[14] = 0.,    fRotM[15] = 1.;
   fInv = b.TestBit(TBuffer3D::kIsReflection);
}

//______________________________________________________________________________
Bool_t TGLTube::IsTransparent()const
{
   return fColor[3] < 1.f;;
}

//______________________________________________________________________________
void TGLTube::GLDraw()const
{
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

   if (GLUquadric *quadObj = GetQuadric()) {
      glLoadName(GetGLName());
      glPushMatrix();

      glTranslated(fX, fY, fZ);
      glMultMatrixd(fRotM);
      glPushMatrix();
      glTranslated(0., 0., -fDz);
      //outer surface
      if (fInv) {
//         glFrontFace(GL_CW);
      }

      gluCylinder(quadObj, fRmax1, fRmax2, 2 * fDz, fNdiv, 1);
      //inner surface
      if (fRmin1 && fRmin2) {
         gluQuadricOrientation(quadObj, (GLenum)GLU_INSIDE);
         gluCylinder(quadObj, fRmin1, fRmin2, 2 * fDz, fNdiv, 1);
         //return orientation back
         gluQuadricOrientation(quadObj, (GLenum)GLU_OUTSIDE);
      }

      glPopMatrix();

      //capping
      glPushMatrix();
      glTranslated(0., 0., fDz);
      gluDisk(quadObj, fRmin2, fRmax2, fNdiv, 1);
      glRotated(180., 1., 0., 0.);
      glTranslated(0., 0., 2 * fDz);
      gluDisk(quadObj, fRmin1, fRmax1, fNdiv, 1);
      glPopMatrix();

      glPopMatrix();

      if (fInv) {
//         glFrontFace(GL_CCW);
      }
   }

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }
}

//______________________________________________________________________________
void TGLTube::Shift(Double_t x, Double_t y, Double_t z)
{
   fX += x;
   fY += y;
   fZ += z;
}

//______________________________________________________________________________
TGLSimpleLight::TGLSimpleLight(UInt_t n, UInt_t l, const Float_t *c, const Double_t *pos)
                   :TGLSceneObject(pos, pos + 3, c, n, 0),
                    fLightName(l)
{
   fColor[16] = -10.f;
   fColor[0] = c[0];
   fColor[1] = c[1];
   fColor[2] = c[2];
   fBulbRad = 10.f;
}

//______________________________________________________________________________
void TGLSimpleLight::GLDraw()const
{
   const Float_t nullColor[] = {0.f, 0.f, 0.f, 1.f};
   const Float_t lightPos[] = {Float_t(fVertices[0]), Float_t(fVertices[1]),
                               Float_t(fVertices[2]), 1.f};
   glMaterialfv(GL_FRONT, GL_EMISSION, fColor);
   glMaterialfv(GL_FRONT, GL_AMBIENT, nullColor);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, nullColor);
   glMaterialfv(GL_FRONT, GL_SPECULAR, nullColor);
   glLightfv(gLightNames[fLightName], GL_DIFFUSE, fColor);
   glLightfv(gLightNames[fLightName], GL_AMBIENT, fColor + 4);
   glLightfv(gLightNames[fLightName], GL_SPECULAR, fColor + 8);
   glLightfv(gLightNames[fLightName], GL_POSITION, lightPos);
   //Draw light source as sphere
   glLoadName(GetGLName());
   glPushMatrix();

   glTranslatef(lightPos[0], lightPos[1], lightPos[2]);
   GLUquadric *quadObj = GetQuadric();
   if (quadObj) gluSphere(quadObj, fBulbRad, 10, 10);

   glPopMatrix();
}

//______________________________________________________________________________
void TGLSimpleLight::Shift(Double_t x, Double_t y, Double_t z)
{
   fVertices[0] += x;
   fVertices[1] += y;
   fVertices[2] += z;
}

//______________________________________________________________________________
void TGLSimpleLight::SetBulbRad(Float_t newRad)
{
   fBulbRad = newRad;
}
