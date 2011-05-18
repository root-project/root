// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGeoPolyShape.h"
#include "TEveGeoShape.h"
#include "TEvePad.h"
#include "TEveUtil.h"

#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TGLScenePad.h"
#include "TGLFaceSet.h"

#include "TList.h"
#include "TGeoBoolNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoMatrix.h"

//______________________________________________________________________________
// Description of TEveGeoPolyShape
//

ClassImp(TEveGeoPolyShape);

//______________________________________________________________________________
TEveGeoPolyShape::TEveGeoPolyShape() :
   TGeoBBox(),
   fNbPols(0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveGeoPolyShape* TEveGeoPolyShape::Construct(TGeoCompositeShape *cshape, Int_t n_seg)
{
   // Static constructor from a composite shape.

   TEvePad       pad;
   TEvePadHolder gpad(kFALSE, &pad);
   TGLScenePad   scene_pad(&pad);
   pad.GetListOfPrimitives()->Add(cshape);
   pad.SetViewer3D(&scene_pad);

   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur(), n_seg);

   scene_pad.BeginScene();
   {
      Double_t halfLengths[3] = { cshape->GetDX(), cshape->GetDY(), cshape->GetDZ() };

      TBuffer3D buff(TBuffer3DTypes::kComposite);
      buff.fID           = cshape;
      buff.fLocalFrame   = kTRUE;
      buff.SetLocalMasterIdentity();
      buff.SetAABoundingBox(cshape->GetOrigin(), halfLengths);
      buff.SetSectionsValid(TBuffer3D::kCore|TBuffer3D::kBoundingBox);

      Bool_t paintComponents = kTRUE;

      // Start a composite shape, identified by this buffer
      if (TBuffer3D::GetCSLevel() == 0)
         paintComponents = gPad->GetViewer3D()->OpenComposite(buff);

      TBuffer3D::IncCSLevel();

      // Paint the boolean node - will add more buffers to viewer
      TGeoHMatrix xxx;
      TGeoMatrix *gst = TGeoShape::GetTransform();
      TGeoShape::SetTransform(&xxx);
      if (paintComponents) cshape->GetBoolNode()->Paint("");
      TGeoShape::SetTransform(gst);
      // Close the composite shape
      if (TBuffer3D::DecCSLevel() == 0)
         gPad->GetViewer3D()->CloseComposite();
   }
   scene_pad.EndScene();
   pad.SetViewer3D(0);

   TGLFaceSet* fs = dynamic_cast<TGLFaceSet*>(scene_pad.FindLogical(cshape));
   if (!fs) {
      ::Warning("TEveGeoPolyShape::Construct", "Failed extracting CSG tesselation for shape '%s'.", cshape->GetName());
      return 0;
   }

   TEveGeoPolyShape *egps = new TEveGeoPolyShape;
   egps->SetFromFaceSet(fs);
   egps->fOrigin[0] = cshape->GetOrigin()[0];
   egps->fOrigin[1] = cshape->GetOrigin()[1];
   egps->fOrigin[2] = cshape->GetOrigin()[2];
   egps->fDX = cshape->GetDX();
   egps->fDY = cshape->GetDY();
   egps->fDZ = cshape->GetDZ();

   return egps;
}

//______________________________________________________________________________
void TEveGeoPolyShape::SetFromFaceSet(TGLFaceSet* fs)
{
   // Set data-members from a face-set.

   fVertices = fs->GetVertices();
   fPolyDesc = fs->GetPolyDesc();
   fNbPols   = fs->GetNbPols();
}

//______________________________________________________________________________
void TEveGeoPolyShape::FillBuffer3D(TBuffer3D& b, Int_t reqSections, Bool_t) const
{
   // Fill the passed buffer 3D.

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

//______________________________________________________________________________
const TBuffer3D& TEveGeoPolyShape::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
   // Fill static buffer 3D.

   static TBuffer3D buf(TBuffer3DTypes::kGeneric);

   FillBuffer3D(buf, reqSections, localFrame);

   return buf;
}

//______________________________________________________________________________
TBuffer3D* TEveGeoPolyShape::MakeBuffer3D() const
{
   // Create buffer 3D and fill it with point/segment/poly data.

   TBuffer3D* buf = new TBuffer3D(TBuffer3DTypes::kGeneric);

   FillBuffer3D(*buf, TBuffer3D::kCore | TBuffer3D::kRawSizes | TBuffer3D::kRaw, kFALSE);

   return buf;
}
