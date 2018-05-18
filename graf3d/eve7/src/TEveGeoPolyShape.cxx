// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Rtypes.h"

#include "ROOT/TEveGeoPolyShape.hxx"
#include "ROOT/TEveGeoShape.hxx"
#include "ROOT/TEveUtil.hxx"
#include "ROOT/TEveCsgOps.hxx"
#include "ROOT/TEveGluTess.hxx"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TList.h"
#include "TGeoBoolNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class TEveGeoPolyShape
\ingroup TEve
Description of TEveGeoPolyShape
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveGeoPolyShape::TEveGeoPolyShape() :
   TGeoBBox(),
   fNbPols(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Static constructor from a composite shape.

TEveGeoPolyShape* TEveGeoPolyShape::Construct(TGeoCompositeShape *cshape, Int_t n_seg)
{
   TEveGeoPolyShape *egps = new TEveGeoPolyShape;
   egps->fOrigin[0] = cshape->GetOrigin()[0];
   egps->fOrigin[1] = cshape->GetOrigin()[1];
   egps->fOrigin[2] = cshape->GetOrigin()[2];
   egps->fDX = cshape->GetDX();
   egps->fDY = cshape->GetDY();
   egps->fDZ = cshape->GetDZ();

   Csg::TBaseMesh *mesh = Csg::BuildFromCompositeShape(cshape, n_seg);
   egps->SetFromMesh(mesh);
   delete mesh;

   return egps;
}

////////////////////////////////////////////////////////////////////////////////
/// Set data-members from a Csg mesh.

void TEveGeoPolyShape::SetFromMesh(Csg::TBaseMesh* mesh)
{
   assert(fNbPols == 0);

   Int_t nv = mesh->NumberOfVertices();
   fVertices.reserve(3 * nv);
   Int_t i;

   for (i = 0; i < nv; ++i)
   {
      const Double_t *v = mesh->GetVertex(i);
      fVertices.insert(fVertices.end(), v, v + 3);
   }

   fNbPols = mesh->NumberOfPolys();

   Int_t descSize = 0;

   for (i = 0; i < fNbPols; ++i) descSize += mesh->SizeOfPoly(i) + 1;

   fPolyDesc.reserve(descSize);

   for (Int_t polyIndex = 0; polyIndex < fNbPols; ++polyIndex)
   {
      Int_t polySize = mesh->SizeOfPoly(polyIndex);

      fPolyDesc.push_back(polySize);

      for (i = 0; i < polySize; ++i) fPolyDesc.push_back(mesh->GetVertexIndex(polyIndex, i));
   }

   // In TGLFaceSet we also did this:
   // if (fgEnforceTriangles)
   // {
   //    EnforceTriangles();
   // }
   // CalculateNormals();
}

void TEveGeoPolyShape::SetFromBuff3D(const TBuffer3D& buffer)
{
   fNbPols = (Int_t) buffer.NbPols();

   if (fNbPols == 0) return;

   Int_t *segs = buffer.fSegs;
   Int_t *pols = buffer.fPols;

   Int_t descSize = 0;

   for (Int_t i = 0, j = 1; i < fNbPols; ++i, ++j)
   {
      descSize += pols[j] + 1;
      j += pols[j] + 1;
   }

   fPolyDesc.resize(descSize);

   for (Int_t numPol = 0, currInd = 0, j = 1; numPol < fNbPols; ++numPol)
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

   if (fEnforceTriangles)  EnforceTriangles();
   if (fCalculateNormals)  CalculateNormals();
}

////////////////////////////////////////////////////////////////////////////////
/// CalculateNormals per polygon (flat shading)

void TEveGeoPolyShape::CalculateNormals()
{
   fNormals.resize(3 * fNbPols);
   if (fNbPols == 0) return;
   Double_t *pnts = &fVertices[0];
   for (Int_t i = 0, j = 0; i < fNbPols; ++i)
   {
      Int_t polEnd = fPolyDesc[j] + j + 1;
      Int_t norm[] = {fPolyDesc[j + 1], fPolyDesc[j + 2], fPolyDesc[j + 3]};
      j += 4;
      Int_t check = CheckPoints(norm, norm);
      Int_t ngood = check;
      if (check == 3) {
         TMath::Normal2Plane(pnts + norm[0] * 3, pnts + norm[1] * 3,
                             pnts + norm[2] * 3, &fNormals[i * 3]);
         j = polEnd;
         continue;
      }
      while (j < polEnd)
      {
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

////////////////////////////////////////////////////////////////////////////////
/// Use GLU tesselator to replace all polygons with N > 3 with triangles.
/// After this call polygon descriptions are changed.
/// New vertices are not expected -- exception is thrown if this is
/// requested by the triangulator. Support for adding of new vertices can be
/// provided.

void TEveGeoPolyShape::EnforceTriangles()
{
   GLU::TriangleCollector tc;

   tc.ProcessData(fVertices, fPolyDesc, fNbPols);

   fPolyDesc.swap(tc.RefPolyDesc());
   fNbPols = tc.GetNTrianlges();
}

////////////////////////////////////////////////////////////////////////////////
/// CheckPoints

Int_t TEveGeoPolyShape::CheckPoints(const Int_t *source, Int_t *dest) const
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

////////////////////////////////////////////////////////////////////////////////
/// Test equality of points with epsilon 1e-10.

Bool_t TEveGeoPolyShape::Eq(const Double_t *p1, const Double_t *p2)
{
   Double_t dx = TMath::Abs(p1[0] - p2[0]);
   Double_t dy = TMath::Abs(p1[1] - p2[1]);
   Double_t dz = TMath::Abs(p1[2] - p2[2]);
   return dx < 1e-10 && dy < 1e-10 && dz < 1e-10;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the passed buffer 3D.

void TEveGeoPolyShape::FillBuffer3D(TBuffer3D& b, Int_t reqSections, Bool_t) const
{
   if (reqSections & TBuffer3D::kCore)
   {
      // If writing core section all others will be invalid
      b.ClearSectionsValid();

      b.fID = const_cast<TEveGeoPolyShape*>(this);
      b.fColor = 0;
      b.fTransparency = 0;
      b.fLocalFrame = kFALSE;
      b.fReflection = kTRUE;

      b.SetSectionsValid(TBuffer3D::kCore);
   }

   if (reqSections & TBuffer3D::kRawSizes || reqSections & TBuffer3D::kRaw)
   {
      Int_t nvrt = fVertices.size() / 3;
      Int_t nseg = 0;

      std::map<Edge_t, Int_t> edges;

      const Int_t *pd = &fPolyDesc[0];
      for (Int_t i = 0; i < fNbPols; ++i)
      {
         Int_t nv = pd[0]; ++pd;
         for (Int_t j = 0; j < nv; ++j)
         {
            Edge_t e(pd[j], (j != nv - 1) ? pd[j+1] : pd[0]);
            if (edges.find(e) == edges.end())
            {
               edges.insert(std::make_pair(e, 0));
               ++nseg;
            }
         }
         pd += nv;
      }

      b.SetRawSizes(nvrt, 3*nvrt, nseg, 3*nseg, fNbPols, fNbPols+fPolyDesc.size());

      memcpy(b.fPnts, &fVertices[0], sizeof(Double_t)*fVertices.size());

      Int_t si = 0, scnt = 0;
      for (std::map<Edge_t, Int_t>::iterator i = edges.begin(); i != edges.end(); ++i)
      {
         b.fSegs[si++] = 0;
         b.fSegs[si++] = i->first.fI;
         b.fSegs[si++] = i->first.fJ;
         i->second = scnt++;
      }

      Int_t pi = 0;
      pd = &fPolyDesc[0];
      for (Int_t i = 0; i < fNbPols; ++i)
      {
         Int_t nv = pd[0]; ++pd;
         b.fPols[pi++] = 0;
         b.fPols[pi++] = nv;
         for (Int_t j = 0; j < nv; ++j)
         {
            b.fPols[pi++] = edges[Edge_t(pd[j], (j != nv - 1) ? pd[j+1] : pd[0])];
         }
         pd += nv;
      }

      b.SetSectionsValid(TBuffer3D::kRawSizes | TBuffer3D::kRaw);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill static buffer 3D.

const TBuffer3D& TEveGeoPolyShape::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
   static TBuffer3D buf(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buf, reqSections, localFrame);

   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Create buffer 3D and fill it with point/segment/poly data.

TBuffer3D* TEveGeoPolyShape::MakeBuffer3D() const
{
   TBuffer3D* buf = new TBuffer3D(TBuffer3DTypes::kGeneric);

   FillBuffer3D(*buf, TBuffer3D::kCore | TBuffer3D::kRawSizes | TBuffer3D::kRaw, kFALSE);

   return buf;
}
