// @(#)root/gl:$Id$
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

#include "TGLFaceSet.h"
#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TMath.h"

// For debug tracing
#include "TClass.h"
#include "TError.h"

#include <stdexcept>

// Clone from TGLUtil -- typedefs needed for portable tesselator function typedef.

#ifndef CALLBACK
#define CALLBACK
#endif

extern "C"
{
#if defined(__APPLE_CC__) && __APPLE_CC__ > 4000 && __APPLE_CC__ < 5450 && !defined(__INTEL_COMPILER)
    typedef GLvoid (*tessfuncptr_t)(...);
#elif defined(__linux__) || defined(__FreeBSD__) || defined( __OpenBSD__ ) || defined(__sun) || defined (__CYGWIN__) || defined (__APPLE__)
    typedef GLvoid (*tessfuncptr_t)();
#elif defined (WIN32)
    typedef GLvoid (CALLBACK *tessfuncptr_t)();
#else
    #error "Error - need to define type tessfuncptr_t for this platform/compiler"
#endif
}

//______________________________________________________________________________
//
// Implementss a native ROOT-GL representation of an arbitrary set of
// polygons.

ClassImp(TGLFaceSet);

Bool_t TGLFaceSet::fgEnforceTriangles = kFALSE;

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buffer) :
   TGLLogicalShape(buffer),
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fNormals(0)
{
   // constructor
   fNbPols = buffer.NbPols();

   if (fNbPols == 0) return;

   Int_t *segs = buffer.fSegs;
   Int_t *pols = buffer.fPols;

   Int_t descSize = 0;

   for (UInt_t i = 0, j = 1; i < fNbPols; ++i, ++j)
   {
      descSize += pols[j] + 1;
      j += pols[j] + 1;
   }

   fPolyDesc.resize(descSize);

   for (UInt_t numPol = 0, currInd = 0, j = 1; numPol < fNbPols; ++numPol)
   {
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

   if (fgEnforceTriangles) {
      EnforceTriangles();
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

      for(i = 0; i < polySize; ++i) fPolyDesc.push_back(mesh->GetVertexIndex(polyIndex, i));
   }

   if (fgEnforceTriangles) {
      EnforceTriangles();
   }
   CalculateNormals();
}

//______________________________________________________________________________
void TGLFaceSet::EnforceTriangles()
{
   // Use GLU tesselator to replace all polygons with N > 3 with triangles.
   // After this call polygon descriptions are changed.
   // New vertices are not expected -- exception is thrown if this is
   // requested by the triangulator. Support for adding of new vertices can be
   // provided.

   class TriangleCollector
   {
   protected:
      Int_t              fNTriangles;
      Int_t              fNVertices;
      Int_t              fV0, fV1;
      GLenum             fType;
      std::vector<Int_t> fPolyDesc;

      void add_triangle(Int_t v0, Int_t v1, Int_t v2)
      {
         fPolyDesc.push_back(3);
         fPolyDesc.push_back(v0);
         fPolyDesc.push_back(v1);
         fPolyDesc.push_back(v2);
         ++fNTriangles;
      }

      void process_vertex(Int_t vi)
      {
         ++fNVertices;

         if (fV0 == -1) {
            fV0 = vi;
            return;
         }
         if (fV1 == -1) {
            fV1 = vi;
            return;
         }

         switch (fType)
         {
            case GL_TRIANGLES:
            {
               add_triangle(fV0, fV1, vi);
               fV0 = fV1 = -1;
               break;
            }
            case GL_TRIANGLE_STRIP:
            {
               if (fNVertices % 2 == 0)
                  add_triangle(fV1, fV0, vi);
               else
                  add_triangle(fV0, fV1, vi);
               fV0 = fV1;
               fV1 = vi;
               break;
            }
            case GL_TRIANGLE_FAN:
            {
               add_triangle(fV0, fV1, vi);
               fV1 = vi;
               break;
            }
            default:
            {
               throw std::runtime_error("TGLFaceSet::EnforceTriangles unexpected type in tess_vertex callback.");
            }
         }
      }

   public:
      TriangleCollector(GLUtesselator* ts) :
         fNTriangles(0), fNVertices(0), fV0(-1), fV1(-1), fType(GL_NONE)
      {
         gluTessCallback(ts, (GLenum)GLU_TESS_BEGIN_DATA,   (tessfuncptr_t) tess_begin);
         gluTessCallback(ts, (GLenum)GLU_TESS_VERTEX_DATA,  (tessfuncptr_t) tess_vertex);
         gluTessCallback(ts, (GLenum)GLU_TESS_COMBINE_DATA, (tessfuncptr_t) tess_combine);
         gluTessCallback(ts, (GLenum)GLU_TESS_END_DATA,     (tessfuncptr_t) tess_end);
      }

      Int_t               GetNTrianlges() { return fNTriangles; }
      std::vector<Int_t>& RefPolyDesc()   { return fPolyDesc; }

      static void tess_begin(GLenum type, TriangleCollector* tc)
      {
         tc->fNVertices = 0;
         tc->fV0 = tc->fV1 = -1;
         tc->fType = type;
      }

      static void tess_vertex(Int_t* vi, TriangleCollector* tc)
      {
         tc->process_vertex(*vi);
      }

      static void tess_combine(GLdouble /*coords*/[3], void* /*vertex_data*/[4],
                               GLfloat  /*weight*/[4], void** /*outData*/,
                               TriangleCollector* /*tc*/)
      {
         throw std::runtime_error("TGLFaceSet::EnforceTriangles tesselator requested vertex combining -- not supported yet.");
      }

      static void tess_end(TriangleCollector* tc)
      {
         tc->fType = GL_NONE;
      }
   };

   GLUtesselator *tess = gluNewTess();
   if (!tess) throw std::bad_alloc();

   TriangleCollector tc(tess);

   // Loop ...
   const Double_t *pnts = &fVertices[0];
   const Int_t    *pols = &fPolyDesc[0];

   for (UInt_t i = 0, j = 0; i < fNbPols; ++i)
   {
      Int_t npoints = pols[j++];

      gluTessBeginPolygon(tess, &tc);
      gluTessBeginContour(tess);

      for (Int_t k = 0; k < npoints; ++k, ++j)
      {
         gluTessVertex(tess, (Double_t*) pnts + pols[j] * 3, (GLvoid*) &pols[j]);
      }

      gluTessEndContour(tess);
      gluTessEndPolygon(tess);
   }

   gluDeleteTess(tess);

   fPolyDesc.swap(tc.RefPolyDesc());
   fNbPols = tc.GetNTrianlges();
}

//______________________________________________________________________________
void TGLFaceSet::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLFaceSet::DirectDraw", "this %ld (class %s) LOD %d", (Long_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   if (fNbPols == 0) return;

   GLUtesselator  *tessObj = TGLUtil::GetDrawTesselator3dv();
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
   // CheckPoints
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
   // test equality
   Double_t dx = TMath::Abs(p1[0] - p2[0]);
   Double_t dy = TMath::Abs(p1[1] - p2[1]);
   Double_t dz = TMath::Abs(p1[2] - p2[2]);
   return dx < 1e-10 && dy < 1e-10 && dz < 1e-10;
}

//______________________________________________________________________________
void TGLFaceSet::CalculateNormals()
{
   // CalculateNormals

   fNormals.resize(3 *fNbPols);
   if (fNbPols == 0) return;
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
Bool_t TGLFaceSet::GetEnforceTriangles()
{
   // Get current state of static flag EnforceTriangles.

   return fgEnforceTriangles;
}

//______________________________________________________________________________
void TGLFaceSet::SetEnforceTriangles(Bool_t e)
{
   // Set state of static flag EnforceTriangles.
   // When this is set, all tesselations will be automatically converted into
   // triangle-only meshes.
   // This is needed to export TGeo shapes and CSG meshes to external
   // triangle-mesh libraries that can not handle arbitrary polygons.

   fgEnforceTriangles = e;
}
