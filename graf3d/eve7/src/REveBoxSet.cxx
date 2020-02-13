// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "REveBoxSet.h"
#include "REveShape.h"

#include "TRandom.h"

/** \class REveBoxSet
\ingroup REve
Collection of 3D primitives (fixed-size boxes, boxes of different
sizes, or arbitrary sexto-epipeds, cones). Each primitive can be assigned
a signal value and a TRef.

A collection of 3D-markers. The way how they are defined depends
on the fBoxType data-member.
  - kBT_FreeBox         arbitrary box: specify 8*(x,y,z) box corners
  - kBT_AABox           axis-aligned box: specify (x,y,z) and (w, h, d)
  - kBT_AABoxFixedDim   axis-aligned box w/ fixed dimensions: specify (x,y,z)
                         also set fDefWidth, fDefHeight and fDefDepth
  - kBT_Cone            cone defined with position, axis-vector and radius
  - EllipticCone        cone with elliptic base (specify another radius and angle in deg)

Each primitive can be assigned:

  1. Color or signal value. Thresholds and signal-to-color mapping
     can then be set dynamically via the REveRGBAPalette class.
  2. External TObject* (stored as TRef).

See also base-class REveDigitSet for more information.
Tutorial: tutorials/eve/boxset_test.C
*/

ClassImp(REveBoxSet);

////////////////////////////////////////////////////////////////////////////////

REveBoxSet::REveBoxSet(const char* n, const char* t) :
   REveDigitSet  (n, t),

   fBoxType      (kBT_Undef),
   fDefWidth     (1),
   fDefHeight    (1),
   fDefDepth     (1),

   fBoxSkip      (0),

   fDrawConeCap  (kFALSE)
{
   // Constructor.

   // Override from REveDigitSet.
   fDisableLighting = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return size of data-structure describing a box of type bt.

Int_t REveBoxSet::SizeofAtom(REveBoxSet::EBoxType_e bt)
{
   static const REveException eH("REveBoxSet::SizeofAtom ");

   switch (bt) {
      case kBT_Undef:                return 0;
      case kBT_FreeBox:              return sizeof(BFreeBox_t);
      case kBT_AABox:                return sizeof(BAABox_t);
      case kBT_AABoxFixedDim:        return sizeof(BAABoxFixedDim_t);
      case kBT_Cone:                 return sizeof(BCone_t);
      case kBT_EllipticCone:         return sizeof(BEllipticCone_t);
      case kBT_Hex:                  return sizeof(BHex_t);
      default:                       throw(eH + "unexpected atom type.");
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the data containers to zero size.
/// The arguments describe the basic parameters of data storage.

void REveBoxSet::Reset(REveBoxSet::EBoxType_e boxType, Bool_t valIsCol, Int_t chunkSize)
{
   fBoxType      = boxType;
   fValueIsColor = valIsCol;
   fDefaultValue = valIsCol ? 0 : kMinInt;
   if (fOwnIds)
      ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), chunkSize);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the data containers to zero size.
/// Keep the old data-storage parameters.

void REveBoxSet::Reset()
{
   if (fOwnIds)
      ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), TMath::Max(fPlex.N(), 64));
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new box from a set of 8 vertices.
/// To be used for box-type kBT_FreeBox.

void REveBoxSet::AddBox(const Float_t* verts)
{
   static const REveException eH("REveBoxSet::AddBox ");

   if (fBoxType != kBT_FreeBox)
      throw(eH + "expect free box-type.");

   BFreeBox_t* b = (BFreeBox_t*) NewDigit();
   memcpy(b->fVertices, verts, sizeof(b->fVertices));
   REveShape::CheckAndFixBoxOrientationFv(b->fVertices);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box from at a given position and with
/// specified dimensions.
/// To be used for box-type kBT_AABox.

void REveBoxSet::AddBox(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h, Float_t d)
{
   static const REveException eH("REveBoxSet::AddBox ");

   if (fBoxType != kBT_AABox)
      throw(eH + "expect axis-aligned box-type.");

   BAABox_t* box = (BAABox_t*) NewDigit();
   box->fA = a; box->fB = b; box->fC = c;
   box->fW = w; box->fH = h; box->fD = d;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box from at a given position.
/// To be used for box-type kBT_AABoxFixedDim.

void REveBoxSet::AddBox(Float_t a, Float_t b, Float_t c)
{
   static const REveException eH("REveBoxSet::AddBox ");

   if (fBoxType != kBT_AABoxFixedDim)
      throw(eH + "expect axis-aligned fixed-dimension box-type.");

   BAABoxFixedDim_t* box = (BAABoxFixedDim_t*) NewDigit();
   box->fA = a; box->fB = b; box->fC = c;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a cone with apex at pos, axis dir and radius r.
/// To be used for box-type kBT_Cone.

void REveBoxSet::AddCone(const REveVector& pos, const REveVector& dir, Float_t r)
{
   static const REveException eH("REveBoxSet::AddCone ");

   if (fBoxType != kBT_Cone)
      throw(eH + "expect cone box-type.");

   BCone_t* cone = (BCone_t*) NewDigit();
   cone->fPos = pos;
   cone->fDir = dir;
   cone->fR   = r;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a cone with apex at pos, axis dir and radius r.
/// To be used for box-type kBT_EllipticCone.

void REveBoxSet::AddEllipticCone(const REveVector& pos, const REveVector& dir,
                                 Float_t r, Float_t r2, Float_t angle)
{
   static const REveException eH("REveBoxSet::AddEllipticCone ");

   if (fBoxType != kBT_EllipticCone)
      throw(eH + "expect elliptic-cone box-type.");

   BEllipticCone_t* cone = (BEllipticCone_t*) NewDigit();
   cone->fPos = pos;
   cone->fDir = dir;
   cone->fR   = r;
   cone->fR2  = r2;
   cone->fAngle = angle;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a hexagonal prism with center of one hexagon at pos, radius of
/// hexagon vertices r, rotation angle angle (in degrees), and length along z
/// of depth. To be used for box-type kBT_Hex.

void REveBoxSet::AddHex(const REveVector& pos, Float_t r, Float_t angle, Float_t depth)
{
   static const REveException eH("REveBoxSet::AddEllipticCone ");

   if (fBoxType != kBT_Hex)
      throw(eH + "expect hex box-type.");

   BHex_t* hex = (BHex_t*) NewDigit();
   hex->fPos   = pos;
   hex->fR     = r;
   hex->fAngle = angle;
   hex->fDepth = depth;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill bounding-box information of the base-class TAttBBox (virtual method).
/// If member 'REveFrameBox* fFrame' is set, frame's corners are used as bbox.

void REveBoxSet::ComputeBBox()
{
   static const REveException eH("REveBoxSet::ComputeBBox ");

   if (fFrame != 0)
   {
      BBoxInit();
      Int_t    n    = fFrame->GetFrameSize() / 3;
      Float_t *bbps = fFrame->GetFramePoints();
      for (int i=0; i<n; ++i, bbps+=3)
         BBoxCheckPoint(bbps);
      return;
   }

   if(fPlex.Size() == 0)
   {
      BBoxZero();
      return;
   }

   BBoxInit();

   REveChunkManager::iterator bi(fPlex);
   switch (fBoxType)
   {

      case kBT_FreeBox:
      {
         while (bi.next()) {
            BFreeBox_t& b = * (BFreeBox_t*) bi();
            for (Int_t i = 0; i < 8; ++i)
               BBoxCheckPoint(b.fVertices[i]);
         }
         break;
      }

      case kBT_AABox:
      {
         while (bi.next()) {
            BAABox_t& b = * (BAABox_t*) bi();
            BBoxCheckPoint(b.fA, b.fB, b.fC);
            BBoxCheckPoint(b.fA + b.fW, b.fB + b.fH , b.fC + b.fD);
         }
         break;
      }

      case kBT_AABoxFixedDim:
      {
         while (bi.next()) {
            BAABoxFixedDim_t& b = * (BAABoxFixedDim_t*) bi();
            BBoxCheckPoint(b.fA, b.fB, b.fC);
            BBoxCheckPoint(b.fA + fDefWidth, b.fB + fDefHeight , b.fC + fDefDepth);
         }
         break;
      }

      case kBT_Cone:
      {
         Float_t mag2=0, mag2Max=0, rMax=0;
         while (bi.next()) {
            BCone_t& b = * (BCone_t*) bi();
            BBoxCheckPoint(b.fPos.fX, b.fPos.fY, b.fPos.fZ);
            mag2 = b.fDir.Mag2();
            if (mag2>mag2Max) mag2Max=mag2;
            if (b.fR>rMax)    rMax=b.fR;
         }
         Float_t off = TMath::Sqrt(mag2Max + rMax*rMax);
         fBBox[0] -= off;fBBox[2] -= off;fBBox[4] -= off;
         fBBox[1] += off;fBBox[3] += off;fBBox[5] += off;
         break;
      }

      case kBT_EllipticCone:
      {
         Float_t mag2=0, mag2Max=0, rMax=0;
         while (bi.next()) {
            BEllipticCone_t& b = * (BEllipticCone_t*) bi();
            BBoxCheckPoint(b.fPos.fX, b.fPos.fY, b.fPos.fZ);
            mag2 = b.fDir.Mag2();
            if (mag2>mag2Max) mag2Max=mag2;
            if (b.fR  > rMax) rMax = b.fR;
            if (b.fR2 > rMax) rMax = b.fR2;
         }
         Float_t off = TMath::Sqrt(mag2Max + rMax*rMax);
         fBBox[0] -= off;fBBox[2] -= off;fBBox[4] -= off;
         fBBox[1] += off;fBBox[3] += off;fBBox[5] += off;
         break;
      }

      case kBT_Hex:
      {
         while (bi.next()) {
            BHex_t& h = * (BHex_t*) bi();
            BBoxCheckPoint(h.fPos.fX - h.fR, h.fPos.fY - h.fR, h.fPos.fZ);
            BBoxCheckPoint(h.fPos.fX + h.fR, h.fPos.fY - h.fR, h.fPos.fZ);
            BBoxCheckPoint(h.fPos.fX + h.fR, h.fPos.fY + h.fR, h.fPos.fZ);
            BBoxCheckPoint(h.fPos.fX - h.fR, h.fPos.fY + h.fR, h.fPos.fZ);
            BBoxCheckPoint(h.fPos.fX - h.fR, h.fPos.fY - h.fR, h.fPos.fZ + h.fDepth);
            BBoxCheckPoint(h.fPos.fX + h.fR, h.fPos.fY - h.fR, h.fPos.fZ + h.fDepth);
            BBoxCheckPoint(h.fPos.fX + h.fR, h.fPos.fY + h.fR, h.fPos.fZ + h.fDepth);
            BBoxCheckPoint(h.fPos.fX - h.fR, h.fPos.fY + h.fR, h.fPos.fZ + h.fDepth);
         }
         break;
      }

      default:
      {
         throw(eH + "unsupported box-type.");
      }

   } // end switch box-type
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the structure with a random set of boxes.

void REveBoxSet::Test(Int_t nboxes)
{
   Reset(kBT_AABox, kTRUE, nboxes);
   TRandom rnd(0);
   const Float_t origin = 10, size = 2;
   Int_t color;
   for(Int_t i=0; i<nboxes; ++i)
   {
      AddBox(origin * rnd.Uniform(-1, 1),
             origin * rnd.Uniform(-1, 1),
             origin * rnd.Uniform(-1, 1),
             size   * rnd.Uniform(0.1, 1),
             size   * rnd.Uniform(0.1, 1),
             size   * rnd.Uniform(0.1, 1));

      REveUtil::ColorFromIdx(rnd.Integer(256), (UChar_t*)&color);
      DigitValue(color);
   }
}
