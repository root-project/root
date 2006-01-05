// @(#)root/gl:$Name:  $:$Id: TGLSceneObject.cxx,v 1.49 2005/11/29 09:25:51 couet Exp $
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
void TGLFaceSet::SetFromMesh(const RootCsg::BaseMesh *mesh)
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
void TGLFaceSet::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLFaceSet::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
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

//______________________________________________________________________________
void TGLFaceSet::DrawWireFrame(UInt_t) const
{
   //
   const Double_t *pnts = &fVertices[0];
   const Int_t *pols = &fPolyDesc[0];


   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t npoints = pols[j++];

      glBegin(GL_POLYGON);

      for (Int_t k = 0; k < npoints; ++k, ++j)
         glVertex3dv(pnts + pols[j] * 3);

      glEnd();
   }
}

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
void TGLPolyMarker::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyMarker::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
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
void TGLPolyLine::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyLine::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
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
void TGLSphere::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLSphere::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

   if (LOD == 0) {
      glPointSize(fRadius*2.0);
      glBegin(GL_POINTS);
      glVertex3d(0.0, 0.0, 0.0);
      glEnd();
   } else {
      // 4 stack/slice min for gluSphere to work
      UInt_t divisions = LOD;
      if (divisions < 4) {
         divisions = 4;
      }
      gluSphere(GetQuadric(),fRadius, divisions, divisions);
   }
}

////////////////////////////////////////////////////////////
namespace GL{
   struct Vertex3d_t {
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

using GL::Vertex3d_t;

Vertex3d_t lowNormal = {{0., 0., -1.}};
Vertex3d_t highNormal = {{0., 0., 1.}};

class TGLMesh {
protected:
   Double_t fRmin1, fRmax1, fRmin2, fRmax2;
   Double_t fDz;
   //Vertex3d_t fCenter; // All shapes now work in local frame so this is not required - check with Timur
   //normals for top and bottom (for cuts)
   Vertex3d_t fNlow;
   Vertex3d_t fNhigh;

   enum {kLod = 40};

   void GetNormal(const Vertex3d_t &vertex, Vertex3d_t &normal)const;
   Double_t GetZcoord(Double_t x, Double_t y, Double_t z)const;
   const Vertex3d_t &MakeVertex(Double_t x, Double_t y, Double_t z)const;

public:
   TGLMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                   const Vertex3d_t &center, const Vertex3d_t &l = lowNormal,
                   const Vertex3d_t &h = highNormal);
   virtual ~TGLMesh() { }
   virtual void Draw(const Double_t *rot)const = 0;
};

//segment contains 3 quad strips:
//one for inner and outer sides, two for top and bottom
class TubeSegMesh : public TGLMesh {
private:
   Vertex3d_t fMesh[(kLod + 1) * 8 + 8];
   Vertex3d_t fNorm[(kLod + 1) * 8 + 8];

public:
   TubeSegMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
               Double_t phi1, Double_t phi2, const Vertex3d_t &center,
               const Vertex3d_t &l = lowNormal, const Vertex3d_t &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//four quad strips:
//outer, inner, top, bottom
class TubeMesh : public TGLMesh {
private:
   Vertex3d_t fMesh[(kLod + 1) * 8];
   Vertex3d_t fNorm[(kLod + 1) * 8];

public:
   TubeMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
            const Vertex3d_t &center, const Vertex3d_t &l = lowNormal,
            const Vertex3d_t &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle funs
class TCylinderMesh : public TGLMesh {
private:
   Vertex3d_t fMesh[(kLod + 1) * 4 + 2];
   Vertex3d_t fNorm[(kLod + 1) * 4 + 2];

public:
   TCylinderMesh(Double_t r1, Double_t r2, Double_t dz, const Vertex3d_t &center,
                const Vertex3d_t &l = lowNormal, const Vertex3d_t &h = highNormal);

   void Draw(const Double_t *rot)const;
};

//One quad mesh and 2 triangle fans
class TCylinderSegMesh : public TGLMesh {
private:
   Vertex3d_t fMesh[(kLod + 1) * 4 + 10];
   Vertex3d_t fNorm[(kLod + 1) * 4 + 10];

public:
   TCylinderSegMesh(Double_t r1, Double_t r2, Double_t dz, Double_t phi1, Double_t phi2,
                   const Vertex3d_t &center, const Vertex3d_t &l = lowNormal,
                   const Vertex3d_t &h = highNormal);

   void Draw(const Double_t *rot)const;
};


//______________________________________________________________________________
TGLMesh::TGLMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                 const Vertex3d_t & /*c*/, const Vertex3d_t &l, const Vertex3d_t &h)
                     :fRmin1(r1), fRmax1(r2), fRmin2(r3), fRmax2(r4),
                      fDz(dz), fNlow(l), fNhigh(h)
{
   //
}

//______________________________________________________________________________
void TGLMesh::GetNormal(const Vertex3d_t &v, Vertex3d_t &n)const
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
const Vertex3d_t &TGLMesh::MakeVertex(Double_t x, Double_t y, Double_t z)const
{
   //
   static Vertex3d_t vert = {{0., 0., 0.}};
   vert[0] = x;
   vert[1] = y;
   vert[2] = GetZcoord(x, y, z);

   return vert;
}

//______________________________________________________________________________
TubeSegMesh::TubeSegMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                         Double_t phi1, Double_t phi2, const Vertex3d_t &center,
                         const Vertex3d_t &l, const Vertex3d_t &h)
                 :TGLMesh(r1, r2, r3, r4, dz, center, l, h), fMesh(), fNorm()

{
   //
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
   Vertex3d_t norm = {{0., 0., 0.}};

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
void TubeSegMesh::Draw(const Double_t * /*rot*/)const
{
   //Tube segment is drawn as three quad strips
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d_t), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d_t), fNorm[0].fXYZ);
   //3. draw first strip
   glDrawArrays(GL_QUAD_STRIP, 0, 4 * (kLod + 1) + 8);
   //4. draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (kLod + 1) + 8, 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (kLod + 1) + 8, 2 * (kLod + 1));

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TubeMesh::TubeMesh(Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t z,
                      const Vertex3d_t &center, const Vertex3d_t &l, const Vertex3d_t &h)
             :TGLMesh(r1, r2, r3, r4, z, center, l, h), fMesh(), fNorm()
{
   //
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
void TubeMesh::Draw(const Double_t * /*rot*/)const
{
   //Tube is drawn as four quad strips
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d_t), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d_t), fNorm[0].fXYZ);
   //draw outer and inner strips
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 2 * (kLod + 1), 2 * (kLod + 1));
   //draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (kLod + 1), 2 * (kLod + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (kLod + 1), 2 * (kLod + 1));
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TCylinderMesh::TCylinderMesh(Double_t r1, Double_t r2, Double_t dz, const Vertex3d_t &center,
                           const Vertex3d_t &l, const Vertex3d_t &h)
                 :TGLMesh(0., r1, 0., r2, dz, center, l, h), fMesh(), fNorm()
{
   //
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
void TCylinderMesh::Draw(const Double_t * /*rot*/)const
{
   //
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d_t), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d_t), fNorm[0].fXYZ);

   //draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1));
   //draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (kLod + 1), kLod + 2);
   glDrawArrays(GL_TRIANGLE_FAN, 3 * (kLod + 1) + 1, kLod + 2);

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

//______________________________________________________________________________
TCylinderSegMesh::TCylinderSegMesh(Double_t r1, Double_t r2, Double_t dz, Double_t phi1,
                                    Double_t phi2, const Vertex3d_t &center, const Vertex3d_t &l,
                                    const Vertex3d_t &h)
                     :TGLMesh(0., r1, 0., r2, dz, center, l, h), fMesh(), fNorm()
{
   //One quad mesh and two fans
   Double_t delta = (phi2 - phi1) / kLod;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const Vertex3d_t vTop = {{0., 0., fDz}};
   const Vertex3d_t vBot = {{0., 0., - fDz}};

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
   Vertex3d_t norm = {{0., 0., 0.}};

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
void TCylinderSegMesh::Draw(const Double_t * /*rot*/)const
{
   //Cylinder segment is drawn as one quad strip and
   //two triangle fans
   //1. enabling vertex arrays
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(Vertex3d_t), fMesh[0].fXYZ);
   glNormalPointer(GL_DOUBLE, sizeof(Vertex3d_t), fNorm[0].fXYZ);
   //3. draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (kLod + 1) + 8);
   //4. draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (kLod + 1) + 8, kLod + 2);
   //      glDrawArrays(GL_TRIANGLE_FAN, 3 * (kLod + 1) + 9, kLod + 2);
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

ClassImp(TGLCylinder)

ClassImp(TGLCylinder)

//______________________________________________________________________________
TGLCylinder::TGLCylinder(const TBuffer3DTube &buffer, TObject *r)
            :TGLSceneObject(buffer, 16, r)
{
   //
   CreateParts(buffer);
}

//______________________________________________________________________________
TGLCylinder::~TGLCylinder()
{
   //
   for (UInt_t i = 0; i < fParts.size(); ++i) {
      delete fParts[i];
      fParts[i] = 0;//not to have invalid pointer for pseudo-destructor call :)
   }
}

//______________________________________________________________________________
void TGLCylinder::CreateParts(const TBuffer3DTube &buffer)
{
   //
   Double_t r1 = buffer.fRadiusInner;
   Double_t r2 = buffer.fRadiusOuter;
   Double_t r3 = buffer.fRadiusInner;
   Double_t r4 = buffer.fRadiusOuter;
   Double_t dz = buffer.fHalfLength;

   // TODO: Check with Timur if this is still required - seems not...?
   const Double_t * lm = buffer.fLocalMaster;
   Vertex3d_t center = {{lm[12], lm[13], lm[14]}};
   Vertex3d_t lowPlaneNorm = {{0., 0., -1.}};
   Vertex3d_t highPlaneNorm = {{0., 0., 1.}};

   switch (buffer.Type()) {
   case TBuffer3DTypes::kTube:
      {
         fParts.push_back(new TubeMesh(r1, r2, r3, r4, dz, center, lowPlaneNorm, highPlaneNorm));
      }
      break;
   case TBuffer3DTypes::kTubeSeg:
   case TBuffer3DTypes::kCutTube:
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

         if (buffer.Type() == TBuffer3DTypes::kCutTube) {
            const TBuffer3DCutTube * cutBuffer = dynamic_cast<const TBuffer3DCutTube *>(&buffer);
            if (!cutBuffer) {
               assert(kFALSE);
               return;
            }

            for (UInt_t i =0; i < 3; i++) {
               lowPlaneNorm[i] = cutBuffer->fLowPlaneNorm[i];
               highPlaneNorm[i] = cutBuffer->fHighPlaneNorm[i];
            }
         }
         fParts.push_back(new TubeSegMesh(r1, r2, r3, r4, dz, phi1,
                                          phi2, center, lowPlaneNorm, highPlaneNorm));
      }
      break;
   default:;
   //polycone should be here
   }
}

//______________________________________________________________________________
void TGLCylinder::DirectDraw(UInt_t LOD) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLCylinder::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), LOD);
   }

   //draw here
   for (UInt_t i = 0; i < fParts.size(); ++i) fParts[i]->Draw(&fVertices[0]);
}
