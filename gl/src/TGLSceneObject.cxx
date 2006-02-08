// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.cxx,v 1.51 2006/01/11 13:44:39 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 **********************************************TF***************************/
#include "TGLSceneObject.h"

#include "TGLIncludes.h"
#include "TAttMarker.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TContextMenu.h"
#include "TGLDrawFlags.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

#include <assert.h>

static GLUtriangulatorObj *GetTesselator()
{
   static struct Init {
      Init()
      {
#if defined(R__WIN32)
         typedef void (CALLBACK *tessfuncptr_t)();
#elif defined(R__AIXGCC)
         typedef void (*tessfuncptr_t)(...);
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

ClassImp(TGLSceneObject)

ClassImp(TGLSceneObject)

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const TBuffer3D &buffer, TObject *obj) :
   TGLLogicalShape(reinterpret_cast<ULong_t>(obj)), // TODO: Clean up more
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fRealObject(obj)
{
   // Use the bounding box in buffer if set
   if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      fBoundingBox.Set(buffer.fBBVertex);
   } else {
   // otherwise use the raw points to generate one   
      assert(buffer.SectionsValid(TBuffer3D::kRaw));
      fBoundingBox.SetAligned(buffer.NbPnts(), buffer.fPnts);
   }
}

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve,
                               TObject *obj) :
   TGLLogicalShape(reinterpret_cast<ULong_t>(obj)), // TODO: Clean up more
   fVertices(verticesReserve, 0.),
   fRealObject(obj)
{
   //
   assert(buffer.SectionsValid(TBuffer3D::kBoundingBox));
   fBoundingBox.Set(buffer.fBBVertex);
}

//______________________________________________________________________________
void TGLSceneObject::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   //
   if (fRealObject) {
      menu.Popup(x, y, fRealObject);
   }
}

ClassImp(TGLFaceSet)

ClassImp(TGLFaceSet)

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buff, TObject *realobj)
               :TGLSceneObject(buff, realobj),
                fNormals(3 * buff.NbPols())
{
   //
   fNbPols = buff.NbPols();

   Int_t *segs = buff.fSegs;
   Int_t *pols = buff.fPols;

   Int_t descSize = 0;

   for (UInt_t i = 0, j = 1; i < fNbPols; ++i, ++j)
   {
      descSize += pols[j] + 1;
      j += pols[j] + 1;
   }

   fPolyDesc.resize(descSize);
   {//fix for scope
   for (UInt_t numPol = 0, currInd = 0, j = 1; numPol < fNbPols; ++numPol) {
      Int_t segmentInd = pols[j] + j;
      Int_t segmentCol = pols[j];
      Int_t s1 = pols[segmentInd];
      segmentInd--;
      Int_t s2 = pols[segmentInd];
      segmentInd--;
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

      Int_t end = j + 1;
      for (; segmentInd != end; segmentInd--) {
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
void TGLFaceSet::SetFromMesh(const RootCsg::TBaseMesh *mesh)
{
   // Should only be done on an empty faceset object
   assert(fNbPols == 0);
   
   UInt_t nv = mesh->NumberOfVertices();
   fVertices.reserve(3 * nv);
   fNormals.resize(mesh->NumberOfPolys() * 3);
   UInt_t i;

   for (i = 0; i < nv; ++i) {
      const Double_t *v = mesh->GetVertex(i);
      fVertices.insert(fVertices.end(), v, v + 3);
   }

   fNbPols = mesh->NumberOfPolys();

   UInt_t descSize = 0;

   for (i = 0; i < fNbPols; ++i) descSize += mesh->SizeOfPoly(i) + 1;

   fPolyDesc.reserve(descSize);

   for (UInt_t polyIndex = 0; polyIndex < fNbPols; ++polyIndex) {
      UInt_t polySize = mesh->SizeOfPoly(polyIndex);

      fPolyDesc.push_back(polySize);

      for(UInt_t i = 0; i < polySize; ++i) fPolyDesc.push_back(mesh->GetVertexIndex(polyIndex, i));
   }

   CalculateNormals();
}

//______________________________________________________________________________
void TGLFaceSet::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLFaceSet::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

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

/*
//______________________________________________________________________________
void TGLFaceSet::DrawOutline(UInt_t lod) const
{
   //
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.f, 1.f);

   DirectDraw(lod);

   glDisable(GL_POLYGON_OFFSET_FILL);
   glDisable(GL_LIGHTING);
   glPolygonMode(GL_FRONT, GL_LINE);
   glColor3d(.1, .1, .1);

   DirectDraw(lod);

   glPolygonMode(GL_FRONT, GL_FILL);
   glEnable(GL_LIGHTING);   
}
*/

//______________________________________________________________________________
Int_t TGLFaceSet::CheckPoints(const Int_t *source, Int_t *dest) const
{
   //
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
   //
   Double_t dx = TMath::Abs(p1[0] - p2[0]);
   Double_t dy = TMath::Abs(p1[1] - p2[1]);
   Double_t dz = TMath::Abs(p1[2] - p2[2]);
   return dx < 1e-10 && dy < 1e-10 && dz < 1e-10;
}

//______________________________________________________________________________
void TGLFaceSet::CalculateNormals()
{
   //
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

ClassImp(TGLPolyMarker)

ClassImp(TGLPolyMarker)

//______________________________________________________________________________
TGLPolyMarker::TGLPolyMarker(const TBuffer3D &buffer, TObject *r)
                  :TGLSceneObject(buffer, r),
                   fStyle(7)
{
   //TAttMarker is not TObject descendant, so I need dynamic_cast
   if (TAttMarker *realObj = dynamic_cast<TAttMarker *>(buffer.fID))
      fStyle = realObj->GetMarkerStyle();
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
   Float_t pointSize = 6.f;
   Double_t topRadius = 5.;
   GLUquadric *quadObj = GetQuadric();

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
}

//______________________________________________________________________________
void TGLPolyMarker::DrawStars()const
{
   //
   glDisable(GL_LIGHTING);
   
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

ClassImp(TGLPolyLine)

ClassImp(TGLPolyLine)

//______________________________________________________________________________
TGLPolyLine::TGLPolyLine(const TBuffer3D &buffer, TObject *r)
                :TGLSceneObject(buffer, r)
{
   //
}

//______________________________________________________________________________
void TGLPolyLine::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyLine::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   glBegin(GL_LINE_STRIP);

   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);

   glEnd();
}

ClassImp(TGLSphere)

ClassImp(TGLSphere)

//______________________________________________________________________________
TGLSphere::TGLSphere(const TBuffer3DSphere &buffer, TObject *r)
                :TGLSceneObject(buffer, r)
{
   // Default ctor
   fRadius = buffer.fRadiusOuter;

   // TODO:
   // Support hollow & cut spheres
   // buffer.fRadiusInner;
   // buffer.fThetaMin;
   // buffer.fThetaMax;
   // buffer.fPhiMin;
   // buffer.fPhiMax;
}

//______________________________________________________________________________
void TGLSphere::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLSphere::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   // 4 stack/slice min for gluSphere to work
   UInt_t divisions = flags.LOD();
   if (divisions < 4) {
      divisions = 4;
   }
   gluSphere(GetQuadric(),fRadius, divisions, divisions);
}

TGLVector3 gLowNormalDefault(0., 0., -1.);
TGLVector3 gHighNormalDefault(0., 0., 1.);

class TGLMesh {
protected:
   // active LOD (level of detail) - quality
   UInt_t     fLOD;    

   Double_t fRmin1, fRmax1, fRmin2, fRmax2;
   Double_t fDz;

   //normals for top and bottom (for cuts)
   TGLVector3 fNlow;
   TGLVector3 fNhigh;

   void GetNormal(const TGLVertex3 &vertex, TGLVector3 &normal)const;
   Double_t GetZcoord(Double_t x, Double_t y, Double_t z)const;
   const TGLVertex3 &MakeVertex(Double_t x, Double_t y, Double_t z)const;

public:
   TGLMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
           const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);
   virtual ~TGLMesh() { }
   virtual void Draw(const Double_t *rot)const = 0;
};

//segment contains 3 quad strips:
//one for inner and outer sides, two for top and bottom
class TubeSegMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLDrawFlags::kLODHigh + 1) * 8 + 8];
   TGLVector3 fNorm[(TGLDrawFlags::kLODHigh + 1) * 8 + 8];

public:
   TubeSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
               Double_t phi1, Double_t phi2, const TGLVector3 &l = gLowNormalDefault, 
               const TGLVector3 &h = gHighNormalDefault);

   void Draw(const Double_t *rot)const;
};

//four quad strips:
//outer, inner, top, bottom
class TubeMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLDrawFlags::kLODHigh + 1) * 8];
   TGLVector3 fNorm[(TGLDrawFlags::kLODHigh + 1) * 8];

public:
   TubeMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
            const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle funs
class TCylinderMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLDrawFlags::kLODHigh + 1) * 4 + 2];
   TGLVector3 fNorm[(TGLDrawFlags::kLODHigh + 1) * 4 + 2];

public:
   TCylinderMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz,
                 const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle fans
class TCylinderSegMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLDrawFlags::kLODHigh + 1) * 4 + 10];
   TGLVector3 fNorm[(TGLDrawFlags::kLODHigh + 1) * 4 + 10];

public:
   TCylinderSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz, Double_t phi1, Double_t phi2,
                    const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);
   void Draw(const Double_t *rot)const;
};


//______________________________________________________________________________
TGLMesh::TGLMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                 const TGLVector3 &l, const TGLVector3 &h) : 
   fLOD(LOD),
   fRmin1(r1), fRmax1(r2), fRmin2(r3), fRmax2(r4),
   fDz(dz), fNlow(l), fNhigh(h)
{
   //
}

//______________________________________________________________________________
void TGLMesh::GetNormal(const TGLVertex3 &v, TGLVector3 &n)const
{
   //
   Double_t z = (fRmax1 - fRmax2) / (2 * fDz);
   Double_t mag = TMath::Sqrt(v[0] * v[0] + v[1] * v[1] + z * z);

   n[0] = v[0] / mag;
   n[1] = v[1] / mag;
   n[2] = z / mag;
}

//______________________________________________________________________________
Double_t TGLMesh::GetZcoord(Double_t x, Double_t y, Double_t z)const
{
   //
   Double_t newz = 0;
   if (z < 0) newz = -fDz - (x * fNlow[0] + y * fNlow[1]) / fNlow[2];
   else newz = fDz - (x * fNhigh[0] + y * fNhigh[1]) / fNhigh[2];

   return newz;
}

//______________________________________________________________________________
const TGLVertex3 &TGLMesh::MakeVertex(Double_t x, Double_t y, Double_t z)const
{
   //
   static TGLVertex3 vert(0., 0., 0.);
   vert[0] = x;
   vert[1] = y;
   vert[2] = GetZcoord(x, y, z);

   return vert;
}

//______________________________________________________________________________
TubeSegMesh::TubeSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                         Double_t phi1, Double_t phi2,
                         const TGLVector3 &l, const TGLVector3 &h)
                 :TGLMesh(LOD, r1, r2, r3, r4, dz, l, h), fMesh(), fNorm()

{
   //
   const Double_t delta = (phi2 - phi1) / LOD;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);
   const Int_t topShift = (fLOD + 1) * 4 + 8;
   const Int_t botShift = (fLOD + 1) * 6 + 8;
   Int_t j = 4 * (fLOD + 1) + 2;

   //defining all three strips here, first strip is non-closed here
   for (Int_t i = 0, e = (fLOD + 1) * 2; i < e; ++i) {
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
   Int_t ind = 2 * (fLOD + 1);
   TGLVector3 norm(0., 0., 0.);

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[ind + 4];
   fMesh[ind + 3] = fMesh[ind + 5];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind = topShift - 4;
   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

//______________________________________________________________________________
void TubeSegMesh::Draw(const Double_t * /*rot*/)const
{
   //Tube segment is drawn as three quad strips
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //3. draw first strip
   glDrawArrays(GL_QUAD_STRIP, 0, 4 * (fLOD + 1) + 8);
   //4. draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (fLOD + 1) + 8, 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (fLOD + 1) + 8, 2 * (fLOD + 1));

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TubeMesh::TubeMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t z,
                   const TGLVector3 &l, const TGLVector3 &h)
             :TGLMesh(LOD, r1, r2, r3, r4, z, l, h), fMesh(), fNorm()
{
   //
   const Double_t delta = TMath::TwoPi() / fLOD;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const Int_t topShift = (fLOD + 1) * 4;
   const Int_t botShift = (fLOD + 1) * 6;
   Int_t j = 4 * (fLOD + 1) - 2;

   //defining all four strips here
   for (Int_t i = 0, e = (fLOD + 1) * 2; i < e; ++i) {
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
void TubeMesh::Draw(const Double_t * /*rot*/)const
{
   //Tube is drawn as four quad strips
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //draw outer and inner strips
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 2 * (fLOD + 1), 2 * (fLOD + 1));
   //draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (fLOD + 1), 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (fLOD + 1), 2 * (fLOD + 1));
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TCylinderMesh::TCylinderMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz,
                             const TGLVector3 &l, const TGLVector3 &h)
                 :TGLMesh(LOD, 0., r1, 0., r2, dz, l, h), fMesh(), fNorm()
{
   //
   const Double_t delta = TMath::TwoPi() / fLOD;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   //central point of top fan
   Int_t topShift = (fLOD + 1) * 2;
   fMesh[topShift][0] = fMesh[topShift][1] = 0., fMesh[topShift][2] = fDz;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //central point of bottom fun
   Int_t botShift = topShift + 2 * (fLOD + 1);
   fMesh[botShift][0] = fMesh[botShift][1] = 0., fMesh[botShift][2] = -fDz;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining 1 strip and 2 fans
   for (Int_t i = 0, e = (fLOD + 1) * 2, j = 0; i < e; ++i) {
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
void TCylinderMesh::Draw(const Double_t * /*rot*/)const
{
   //
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());

   //draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1));
   //draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (fLOD + 1), fLOD + 2);
   glDrawArrays(GL_TRIANGLE_FAN, 3 * (fLOD + 1) + 1, fLOD + 2);

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TCylinderSegMesh::TCylinderSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz, Double_t phi1,
                                    Double_t phi2, const TGLVector3 &l,
                                    const TGLVector3 &h)
                     :TGLMesh(LOD, 0., r1, 0., r2, dz, l, h), fMesh(), fNorm()
{
   //One quad mesh and two fans
   Double_t delta = (phi2 - phi1) / fLOD;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const TGLVertex3 vTop(0., 0., fDz);
   const TGLVertex3 vBot(0., 0., - fDz);

   //center of top fan
   Int_t topShift = (fLOD + 1) * 2 + 8;
   fMesh[topShift] = vTop;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //center of bottom fan
   Int_t botShift = topShift + fLOD + 1;
   fMesh[botShift] = vBot;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining strip and two fans
   //strip is not closed here
   Int_t i = 0;
   for (Int_t e = (fLOD + 1) * 2, j = 0; i < e; ++i) {
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
   Int_t ind = 2 * (fLOD + 1);
   TGLVector3 norm(0., 0., 0.);

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = vTop;
   fMesh[ind + 3] = vBot;
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                          norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind += 4;
   fMesh[ind] = vTop;
   fMesh[ind + 1] = vBot;
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

//______________________________________________________________________________
void TCylinderSegMesh::Draw(const Double_t * /*rot*/)const
{
   //Cylinder segment is drawn as one quad strip and
   //two triangle fans
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //3. draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1) + 8);
   //4. draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (fLOD + 1) + 8, fLOD + 2);
   //      glDrawArrays(GL_TRIANGLE_FAN, 3 * (fLOD + 1) + 9, fLOD + 2);
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

ClassImp(TGLCylinder)

//______________________________________________________________________________
TGLCylinder::TGLCylinder(const TBuffer3DTube &buffer, TObject *r)
   :TGLSceneObject(buffer, 16, r)
{
   // Copy out relevant parts of buffer - we create and delete mesh
   // parts on demand in DirectDraw() and they are DL cached
   fR1 = buffer.fRadiusInner;
   fR2 = buffer.fRadiusOuter;
   fR3 = buffer.fRadiusInner;
   fR4 = buffer.fRadiusOuter;
   fDz = buffer.fHalfLength;

   fLowPlaneNorm = gLowNormalDefault; 
   fHighPlaneNorm = gHighNormalDefault; 

   switch (buffer.Type()) {
      case TBuffer3DTypes::kTube:
         fSegMesh = kFALSE;
         break;
   case TBuffer3DTypes::kTubeSeg:
   case TBuffer3DTypes::kCutTube:
         fSegMesh = kTRUE;

         const TBuffer3DTubeSeg * segBuffer = dynamic_cast<const TBuffer3DTubeSeg *>(&buffer);
         if (!segBuffer) {
            Error("TGLCylinder::TGLCylinder", "cannot cast TBuffer3D");
            return;
         }

         fPhi1 = segBuffer->fPhiMin;
         fPhi2 = segBuffer->fPhiMax;
         if (fPhi2 < fPhi1) fPhi2 += 360.;
         fPhi1 *= TMath::DegToRad();
         fPhi2 *= TMath::DegToRad();

         if (buffer.Type() == TBuffer3DTypes::kCutTube) {
            const TBuffer3DCutTube * cutBuffer = dynamic_cast<const TBuffer3DCutTube *>(&buffer);
            if (!cutBuffer) {
               Error("TGLCylinder::TGLCylinder", "cannot cast TBuffer3D");
               return;
            }

            for (UInt_t i =0; i < 3; i++) {
               fLowPlaneNorm[i] = cutBuffer->fLowPlaneNorm[i];
               fHighPlaneNorm[i] = cutBuffer->fHighPlaneNorm[i];
            }
         }
   }
}

//______________________________________________________________________________
TGLCylinder::~TGLCylinder()
{
}

//______________________________________________________________________________
void TGLCylinder::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLCylinder::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   // As we are now support display list caching we can create, draw and
   // delete mesh parts of suitible LOD (quality) here - it will be cached 
   // into a display list by TGLDisplayListCache/TGLDrawable base,
   // against our id and the LOD value. So this will only occur once
   // for a certain cylinder/LOD combination
   std::vector<TGLMesh *> meshParts;

   // Create mesh parts
   if (!fSegMesh) {
      meshParts.push_back(new TubeMesh(flags.LOD(), fR1, fR2, fR3, fR4, fDz, fLowPlaneNorm, fHighPlaneNorm));
   } else {
      meshParts.push_back(new TubeSegMesh(flags.LOD(), fR1, fR2, fR3, fR4, fDz, fPhi1,
                                          fPhi2, fLowPlaneNorm, fHighPlaneNorm));
   }

   // Draw mesh parts
   for (UInt_t i = 0; i < meshParts.size(); ++i) meshParts[i]->Draw(&fVertices[0]);

   // Delete mesh parts
   for (UInt_t i = 0; i < meshParts.size(); ++i) {
      delete meshParts[i];
      meshParts[i] = 0;//not to have invalid pointer for pseudo-destructor call :)
   }
}
