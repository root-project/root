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
#include "TGLFaceSet.h"

#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

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

      const Int_t *pd = &fPolyDesc[0];
      for (UInt_t i = 0; i < fNbPols; ++i)
      {
         nseg += pd[0];
         pd   += pd[0] + 1;
      }

      b.SetRawSizes(nvrt, 3*nvrt, nseg, 3*nseg, fNbPols, fNbPols+fPolyDesc.size());

      memcpy(b.fPnts, &fVertices[0], sizeof(Double_t)*fVertices.size());

      Int_t si = 0, pi = 0, ns = 0;

      pd = &fPolyDesc[0];
      for (UInt_t i = 0; i < fNbPols; ++i)
      {
         UInt_t nv = pd[0]; ++pd;
         b.fPols[pi++] = 0;
         b.fPols[pi++] = nv;
         for (UInt_t j = 0; j < nv; ++j)
         {
            b.fSegs[si++] = 0;
            b.fSegs[si++] = pd[j];
            b.fSegs[si++] = (j != nv - 1) ? pd[j+1] : pd[0];

            b.fPols[pi++] = ns++;
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
