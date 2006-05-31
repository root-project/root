// @(#)root/gl:$Name:  $:$Id: TGLFaceSet.cxx,v 1.1 2006/02/20 11:10:06 brun Exp $
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
#include "TGLDrawFlags.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

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

ClassImp(TGLFaceSet)

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buffer) :
   TGLLogicalShape(buffer),
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fNormals(3 * buffer.NbPols())
{
   // constructor
   fNbPols = buffer.NbPols();

   Int_t *segs = buffer.fSegs;
   Int_t *pols = buffer.fPols;

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
