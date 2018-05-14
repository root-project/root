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
// #include "TEvePad.h"
#include "ROOT/TEveUtil.hxx"
#include "ROOT/TEveCsgOps.hxx"

#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
// #include "TGLScenePad.h"
// #include "TGLFaceSet.h"

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

   UInt_t nv = mesh->NumberOfVertices();
   fVertices.reserve(3 * nv);
   UInt_t i;

   for (i = 0; i < nv; ++i)
   {
      const Double_t *v = mesh->GetVertex(i);
      fVertices.insert(fVertices.end(), v, v + 3);
   }

   fNbPols = mesh->NumberOfPolys();

   UInt_t descSize = 0;

   for (i = 0; i < fNbPols; ++i) descSize += mesh->SizeOfPoly(i) + 1;

   fPolyDesc.reserve(descSize);

   for (UInt_t polyIndex = 0; polyIndex < fNbPols; ++polyIndex)
   {
      UInt_t polySize = mesh->SizeOfPoly(polyIndex);

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
      UInt_t nvrt = fVertices.size() / 3;
      UInt_t nseg = 0;

      std::map<Edge_t, Int_t> edges;

      const Int_t *pd = &fPolyDesc[0];
      for (UInt_t i = 0; i < fNbPols; ++i)
      {
         UInt_t nv = pd[0]; ++pd;
         for (UInt_t j = 0; j < nv; ++j)
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
      for (UInt_t i = 0; i < fNbPols; ++i)
      {
         UInt_t nv = pd[0]; ++pd;
         b.fPols[pi++] = 0;
         b.fPols[pi++] = nv;
         for (UInt_t j = 0; j < nv; ++j)
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
