// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBoxSet.h"
#include "TRandom.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"

//______________________________________________________________________________
// TEveBoxSet
//
// Collection of 3D primitives (fixed-size boxes, boxes of different
// sizes, or arbitrary sexto-epipeds); each primitive can be assigned
// a signal value and a TRef.
//
// A collection of 3D-boxes. The way how the boxes are defined depends
// on the fBoxType data-member.
//   kBT_FreeBox         arbitrary box: specify 8*(x,y,z) box corners
//   kBT_AABox           axis-aligned box: specify (x,y,z) and (w, h, d)
//   kBT_AABoxFixedDim   axis-aligned box w/ fixed dimensions: specify (x,y,z)
//                      also set fDefWidth, fDefHeight and fDefDepth
//
// Each box can be assigned:
// a) Color or signal value. Thresholds and signal-to-color mapping
//    can then be set dynamically via the TEveRGBAPalette class.
// b) External TObject* (stored as TRef).
//
// See also base-class TEveDigitSet for more information.

ClassImp(TEveBoxSet);

//______________________________________________________________________________
TEveBoxSet::TEveBoxSet(const Text_t* n, const Text_t* t) :
   TEveDigitSet  (n, t),

   fBoxType      (kBT_Undef),
   fDefWidth     (1),
   fDefHeight    (1),
   fDefDepth     (1)
{
   // Constructor.

   // Override from TEveDigitSet.
   fDisableLigting = kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
Int_t TEveBoxSet::SizeofAtom(TEveBoxSet::EBoxType_e bt)
{
   // Return size of data-structure describing a box of type bt.

   static const TEveException eH("TEveBoxSet::SizeofAtom ");

   switch (bt) {
      case kBT_Undef:                return 0;
      case kBT_FreeBox:              return sizeof(BFreeBox_t);
      case kBT_AABox:                return sizeof(BAABox_t);
      case kBT_AABoxFixedDim:        return sizeof(BAABoxFixedDim_t);
      default:                      throw(eH + "unexpected atom type.");
   }
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBoxSet::Reset(TEveBoxSet::EBoxType_e boxType, Bool_t valIsCol, Int_t chunkSize)
{
   // Reset the data containers to zero size.
   // The arguments describe the basic parameters of data storage.

   fBoxType      = boxType;
   fValueIsColor = valIsCol;
   fDefaultValue = valIsCol ? 0 : kMinInt;
   if (fOwnIds)
      ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), chunkSize);
}

//______________________________________________________________________________
void TEveBoxSet::Reset()
{
   // Reset the data containers to zero size.
   // Keep the old data-storage parameters.

   if (fOwnIds)
      ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), TMath::Max(fPlex.N(), 64));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBoxSet::AddBox(const Float_t* verts)
{
   // Create a new box from a set of 8 vertices.
   // To be used for box-type kBT_FreeBox.

   static const TEveException eH("TEveBoxSet::AddBox ");

   if (fBoxType != kBT_FreeBox)
      throw(eH + "expect free box-type.");

   BFreeBox_t* b = (BFreeBox_t*) NewDigit();
   memcpy(b->fVertices, verts, sizeof(b->fVertices));
}

//______________________________________________________________________________
void TEveBoxSet::AddBox(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h, Float_t d)
{
   // Create a new axis-aligned box from at a given position and with
   // specified dimensions.
   // To be used for box-type kBT_AABox.

   static const TEveException eH("TEveBoxSet::AddBox ");

   if (fBoxType != kBT_AABox)
      throw(eH + "expect axis-aligned box-type.");

   BAABox_t* box = (BAABox_t*) NewDigit();
   box->fA = a; box->fB = b; box->fC = c;
   box->fW = w; box->fH = h; box->fD = d;
}

//______________________________________________________________________________
void TEveBoxSet::AddBox(Float_t a, Float_t b, Float_t c)
{
   // Create a new axis-aligned box from at a given position.
   // To be used for box-type kBT_AABoxFixedDim.

   static const TEveException eH("TEveBoxSet::AddBox ");

   if (fBoxType != kBT_AABoxFixedDim)
      throw(eH + "expect axis-aligned fixed-dimension box-type.");

   BAABoxFixedDim_t* box = (BAABoxFixedDim_t*) NewDigit();
   box->fA = a; box->fB = b; box->fC = c;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBoxSet::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

   static const TEveException eH("TEveBoxSet::ComputeBBox ");

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

   TEveChunkManager::iterator bi(fPlex);
   switch (fBoxType)
   {

      case kBT_FreeBox:
      {
         while (bi.next()) {
            BFreeBox_t& b = * (BFreeBox_t*) bi();
            Float_t * p = b.fVertices;
            for(int i=0; i<8; ++i, p+=3)
               BBoxCheckPoint(p);
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

      default:
      {
         throw(eH + "unsupported box-type.");
      }

   } // end switch box-type

   printf("%s BBox is x(%f,%f), y(%f,%f), z(%f,%f)\n", GetName(),
          fBBox[0], fBBox[1], fBBox[2], fBBox[3], fBBox[4], fBBox[5]);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBoxSet::Test(Int_t nboxes)
{
   // Fill the structure with a random set of boxes.

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

      TEveUtil::ColorFromIdx(rnd.Integer(256), (UChar_t*)&color);
      DigitValue(color);
   }
}
