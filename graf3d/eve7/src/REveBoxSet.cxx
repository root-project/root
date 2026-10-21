// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/REveBoxSet.hxx"
#include "ROOT/REveShape.hxx"
#include "ROOT/REveRenderData.hxx"
#include "ROOT/REveRGBAPalette.hxx"
#include "ROOT/REveManager.hxx"
#include "ROOT/REveTrans.hxx"

#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include <cmath>

#include <nlohmann/json.hpp>

using namespace::ROOT::Experimental;

/** \class REveBoxSet
\ingroup REve
Collection of 3D primitives (fixed-size boxes, boxes of different
sizes, or arbitrary sexto-epipeds, cones). Each primitive can be assigned
a signal value and a TRef.

A collection of 3D-markers. The way how they are defined depends
on the fBoxType data-member.
  - kBT_Vertices           arbitrary box: specify its 8 vertices, 8*(x,y,z)
  - kBT_InstancedScaled    axis-aligned box of its own size
  - kBT_Instanced          axis-aligned box of the default size, set by
                            SetDefWidth() & co., or SetDefHalfWidth() & co.
  - kBT_InstancedScaledRotated  any transformation of the instance shape, a box,
                            hexagonal prism or cone: AddInstanceMat4(), AddHex(),
                            AddCone(), AddEllipticCone()

An axis-aligned box is stored as its centre and half size, and spans centre -
half size to centre + half size. AddInstance() and AddInstanceScaled() take the
centre and the half size. AddInstanceCorner() and AddInstanceScaledCorner() take
the lower corner and the full size, as TEveBoxSet's AddBox() does. The default
size of a kBT_Instanced box is set as the full size with SetDefWidth(),
SetDefHeight() and SetDefDepth(), or as the half size with SetDefHalfWidth(),
SetDefHalfHeight() and SetDefHalfDepth(). It defaults to 1, a unit box, as in
TEveBoxSet.

Each primitive can be assigned:

  1. Color or signal value. Thresholds and signal-to-color mapping
     can then be set dynamically via the REveRGBAPalette class.
  2. External TObject* (stored as TRef).

See also base-class REveDigitSet for more information.
Tutorial: tutorials/visualisation/eve/boxset_test.C
*/

////////////////////////////////////////////////////////////////////////////////

REveBoxSet::REveBoxSet(const char* n, const char* t) :
   REveDigitSet  (n, t),

   fBoxType      (kBT_Undef),

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
      case kBT_Undef:                  return 0;
      case kBT_Vertices: return sizeof(Vertices_t);
      case kBT_Instanced:              return sizeof(Instanced_t);
      case kBT_InstancedScaled:         return sizeof(InstancedScaled_t);
      case kBT_InstancedScaledRotated:  return sizeof(InstancedScaledRotated_t);
      default:                        throw(eH + "unexpected atom type.");
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Half size of an axis-aligned instance, kBT_Instanced or kBT_InstancedScaled,
/// as the client draws it. It can be negative where the size is.

void REveBoxSet::InstanceHalfSize(const DigitBase_t &digit, Float_t half[3]) const
{
   if (fBoxType == kBT_InstancedScaled) {
      const InstancedScaled_t &b = (const InstancedScaled_t &)digit;
      half[0] = b.fW;
      half[1] = b.fH;
      half[2] = b.fD;
   } else {
      half[0] = fDefHalfWidth;
      half[1] = fDefHalfHeight;
      half[2] = fDefHalfDepth;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the data containers to zero size.
/// The arguments describe the basic parameters of data storage.

void REveBoxSet::Reset(REveBoxSet::EBoxType_e boxType, Bool_t valIsCol, Int_t chunkSize)
{
   fBoxType      = boxType;
   fValueIsColor = valIsCol;
   fDefaultValue = valIsCol ? 0 : kMinInt;
   ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), chunkSize);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset the data containers to zero size.
/// Keep the old data-storage parameters.

void REveBoxSet::Reset()
{
   ReleaseIds();
   fPlex.Reset(SizeofAtom(fBoxType), TMath::Max(fPlex.N(), 64));
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new box from a set of 8 vertices.
/// To be used for box-type kBT_Vertices.

void REveBoxSet::AddVertices(const Float_t *verts)
{
   static const REveException eH("REveBoxSet::AddVertices ");

   if (fBoxType != kBT_Vertices)
      throw(eH + "expect box-type kBT_Vertices.");

   Vertices_t *b = (Vertices_t *)NewDigit();
   memcpy(b->fVertices, verts, sizeof(b->fVertices));
   REveShape::CheckAndFixBoxOrientationFv(b->fVertices);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box with centre (cx, cy, cz) and half size
/// (hw, hh, hd).
/// To be used for box-type kBT_InstancedScaled.

void REveBoxSet::AddInstanceScaled(Float_t cx, Float_t cy, Float_t cz, Float_t hw, Float_t hh, Float_t hd)
{
   static const REveException eH("REveBoxSet::AddInstanceScaled ");

   if (fBoxType != kBT_InstancedScaled)
      throw(eH + "expect box-type kBT_InstancedScaled.");

   InstancedScaled_t* box = (InstancedScaled_t*) NewDigit();
   box->fX = cx;
   box->fY = cy;
   box->fZ = cz;
   box->fW = hw;
   box->fH = hh;
   box->fD = hd;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box with lower corner (x, y, z) and size
/// (w, h, d).
/// To be used for box-type kBT_InstancedScaled.

void REveBoxSet::AddInstanceScaledCorner(Float_t x, Float_t y, Float_t z, Float_t w, Float_t h, Float_t d)
{
   AddInstanceScaled(x + 0.5f * w, y + 0.5f * h, z + 0.5f * d, 0.5f * w, 0.5f * h, 0.5f * d);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box with centre (cx, cy, cz) and the default
/// size, see SetDefWidth() and SetDefHalfWidth().
/// To be used for box-type kBT_Instanced.

void REveBoxSet::AddInstance(Float_t cx, Float_t cy, Float_t cz)
{
   static const REveException eH("REveBoxSet::AddInstance ");

   if (fBoxType != kBT_Instanced)
      throw(eH + "expect box-type kBT_Instanced.");

   Instanced_t* box = (Instanced_t*) NewDigit();
   box->fX = cx;
   box->fY = cy;
   box->fZ = cz;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new axis-aligned box with lower corner (x, y, z) and the default
/// size, see SetDefWidth() and SetDefHalfWidth(). The box is stored by its
/// centre, computed with the current default size, so set that first.
/// To be used for box-type kBT_Instanced.

void REveBoxSet::AddInstanceCorner(Float_t x, Float_t y, Float_t z)
{
   AddInstance(x + fDefHalfWidth, y + fDefHalfHeight, z + fDefHalfDepth);
}

////////////////////////////////////////////////////////////////////////////////
/// Create shape with arbitrary transformtaion
///
void REveBoxSet::AddInstanceMat4(const Float_t* arr)
{
   static const REveException eH("REveBoxSet::AddInstanceMat4 ");
   if (fBoxType != kBT_InstancedScaledRotated)
      throw(eH + "expect box-type kBT_InstancedScaledRotated.");

   InstancedScaledRotated_t* b = (InstancedScaledRotated_t*) NewDigit();
   memcpy(b->fMat, arr, sizeof(b->fMat));
}

////////////////////////////////////////////////////////////////////////////////
/// Create a cone with apex at pos, axis dir and radius r.
/// To be used for box-type kBT_InstancedScaledRotated.

void REveBoxSet::AddCone(const REveVector& pos, const REveVector& dir, Float_t r)
{
   static const REveException eH("REveBoxSet::AddCone ");
   using namespace TMath;
   if (fBoxType != kBT_InstancedScaledRotated)
      throw(eH + "expect box-type kBT_InstancedScaledRotated.");
   fInstanceShape = kIS_Cone;

   REveTrans t;
   float  h = dir.Mag();
   float phi   = ATan2(dir.fY, dir.fX);
   float theta = ATan (dir.fZ / Sqrt(dir.fX*dir.fX + dir.fY*dir.fY));

   theta =  Pi()/2 -theta;
   t.RotateLF(1, 2, phi);
   t.RotateLF(3, 1, theta);
   t.SetScale(r, r, h);
   t.SetPos(pos.fX, pos.fY, pos.fZ);

   InstancedScaledRotated_t* cone = (InstancedScaledRotated_t*) NewDigit();
   for(Int_t i=0; i<16; ++i)
   cone->fMat[i] = t[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Create a cone with apex at pos, axis dir and radius r.
/// To be used for box-type kBT_EllipticCone.

void REveBoxSet::AddEllipticCone(const REveVector& pos, const REveVector& dir,
                                 Float_t r, Float_t r2, Float_t angle)
{
   static const REveException eH("REveBoxSet::AddEllipticCone ");
   using namespace TMath;
   if (fBoxType != kBT_InstancedScaledRotated)
      throw(eH + "expect box-type kBT_InstancedScaledRotated.");
   fInstanceShape = kIS_Cone;
   REveTrans t;
   float  h = dir.Mag();
   float phi   = ATan2(dir.fY, dir.fX);
   float theta = ATan (dir.fZ / Sqrt(dir.fX*dir.fX + dir.fY*dir.fY));

   theta =  Pi()/2 -theta;
   t.RotateLF(1, 2, phi);
   t.RotateLF(3, 1, theta);
   t.RotateLF(1, 2, angle * TMath::DegToRad());
   t.SetScale(r, r2, h);
   t.SetPos(pos.fX, pos.fY, pos.fZ);

   InstancedScaledRotated_t* cone = (InstancedScaledRotated_t*) NewDigit();
   for(Int_t i=0; i<16; ++i)
   cone->fMat[i] = t[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Create a hexagonal prism with center of one hexagon at pos, radius of
/// hexagon vertices r, rotation angle angle (in degrees), and length along z
/// of depth. To be used for box-type kBT_Hex.

void REveBoxSet::AddHex(const REveVector& pos, Float_t r, Float_t angle, Float_t depth)
{
   static const REveException eH("REveBoxSet::AddHex ");

   if (fBoxType != kBT_InstancedScaledRotated)
      throw(eH + "expect box-type kBT_InstancedScaledRotated.");

   fInstanceShape = kIS_Hex;

   InstancedScaledRotated_t* hex = (InstancedScaledRotated_t*) NewDigit();
   REveTrans t; // AMT do we need to reuse ???
   t.SetPos(pos.fX, pos.fY, pos.fZ);
   t.SetScale(r,r,depth);
   t.RotateLF(1, 2, angle * TMath::DegToRad());
   for(Int_t i=0; i<16; ++i)
   hex->fMat[i] = t[i];
}
////////////////////////////////////////////////////////////////////////////////
/// Fill the bounding box (virtual from REveElement).
/// If member 'REveFrameBox* fFrame' is set, frame's corners are used as bbox.

void REveBoxSet::ComputeBBox()
{
   static const REveException eH("REveBoxSet::ComputeBBox ");

   if (fFrame != nullptr)
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

   // An instance is drawn at pos + M * v, with v in [-1, 1]^3 and M its half
   // sizes or its matrix. Each instance adds its own axis-aligned box.
   REveChunkManager::iterator bi(fPlex);
   switch (fBoxType) {

   case kBT_Vertices: {
      while (bi.next()) {
         Vertices_t &b = *(Vertices_t *)bi();
         for (Int_t i = 0; i < 8; ++i)
            BBoxCheckPoint(b.fVertices[i]);
      }
      break;
   }

   case kBT_InstancedScaled:
   case kBT_Instanced: {
      Float_t half[3];
      while (bi.next()) {
         Instanced_t &b = *(Instanced_t *)bi();
         InstanceHalfSize(b, half);
         BBoxCheckBox(REveVector(b.fX, b.fY, b.fZ), REveVector(half[0], half[1], half[2]));
      }
      break;
   }

   case kBT_InstancedScaledRotated: {
      while (bi.next()) {
         InstancedScaledRotated_t &b = *(InstancedScaledRotated_t *)bi();
         float *a = b.fMat;
         // Column-major: column j of M is a[4j], a[4j+1], a[4j+2]. M [-1, 1]^3
         // reaches |a[k]| + |a[4+k]| + |a[8+k]| along axis k.
         REveVector h;
         for (Int_t k = 0; k < 3; ++k)
            h[k] = std::abs(a[k]) + std::abs(a[4 + k]) + std::abs(a[8 + k]);
         BBoxCheckBox(REveVector(a[12], a[13], a[14]), h);
      }
      break;
   }

   default: {
      throw eH + "unsupported box-type.";
   }

   } // end switch box-type
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveBoxSet::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   j["boxType"] = int(fBoxType);
   j["shapeType"] = int(fInstanceShape);
   if (fInstanceShape == kIS_Cone) {
      j["coneCap"] = fDrawConeCap;
   }

   j["instanced"] = Instanced();
   if (Instanced())
   {
      int  N = fPlex.N();
      int  N_tex = 0;
      std::string instanceFlag;
      switch (fBoxType)
      {
         case kBT_Instanced:
            instanceFlag = "FixedDimension";
            N_tex = N;
            break;
         case kBT_InstancedScaled:
            instanceFlag = "ScalePerDigit";
            N_tex = 2*N;
            break;
         case kBT_InstancedScaledRotated:
           instanceFlag = "Mat4Trans";
           N_tex = 4*N;
           break;
         default:
           R__LOG_ERROR(REveLog()) << "REveBoxSet::WriteCoreJson Unhandled instancing type.";
      }

      REveRenderData::CalcTextureSize(N_tex, 4, fTexX, fTexY);

      j["N"] = N;
      j["texX"] = fTexX;
      j["texY"] = fTexY;
      j["instanceFlag"] = instanceFlag;
      // The client scales the shape template, which is within [-1, 1]^3, by
      // these: the default half size for kBT_Instanced, 1 for the types that
      // carry their own size or matrix.
      const bool fixed = (fBoxType == kBT_Instanced);
      j["defWidth"] = fixed ? fDefHalfWidth : 1.f;
      j["defHeight"] = fixed ? fDefHalfHeight : 1.f;
      j["defDepth"] = fixed ? fDefHalfDepth : 1.f;

      // printf("TEXTURE SIZE X=%d, Y=%d\n", fTexX, fTexY);
   }

   // AMT:: the base class WroteCoreJson needs to be called after
   // setting the texture value
   Int_t ret = REveDigitSet::WriteCoreJson(j, rnr_offset);
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates 3D point array for rendering.

void REveBoxSet::BuildRenderData()
{
   if (fPlex.Size() == 0)
      return;

   fRenderData = std::make_unique<REveRenderData>("makeBoxSet", fPlex.Size() * 24, 0, fPlex.Size());

   REveChunkManager::iterator bi(fPlex);
   while (bi.next()) {
      REveDigitSet::DigitBase_t *b = (REveDigitSet::DigitBase_t *)bi();
      if (IsDigitVisible(b)) {
         WriteShapeData(*b);
         if (fSingleColor == false) {

            if (fValueIsColor) {
               fRenderData->PushI(int(b->fValue));
            } else {
               UChar_t c[4] = {0, 0, 0, 0};
               fPalette->ColorFromValue(b->fValue, fDefaultValue, c);

               int value = c[0] + c[1] * 256 + c[2] * 256 * 256;
               // printf("box val [%d] values (%d, %d, %d) -> int <%d>\n", b.fValue, c[0], c[1], c[2],  value);
               fRenderData->PushI(value);
            }
         }
      }
   }
   if (Instanced()) {
      // printf(" >>> resize render data %d \n", 4 * fTexX * fTexY);
      fRenderData->ResizeV(4 * fTexX * fTexY);
   }

   REveElement::BuildRenderData();
}

////////////////////////////////////////////////////////////////////////////////
/// Get int value for color. Used for case of instancing.
///
unsigned int REveBoxSet::GetColorFromDigit(REveDigitSet::DigitBase_t &digi)
{
   if (fSingleColor == false) {
      if (fValueIsColor) {
        UChar_t* c = (UChar_t*) & digi.fValue;
        return (c[0] << 16) + (c[1] << 8) + c[2];
      } else {
         // printf("palette\n");
         UChar_t c[4] = {0, 0, 0, 0};
         fPalette->ColorFromValue(digi.fValue, fDefaultValue, c);
         return (c[0] << 16) + (c[1] << 8) + c[2];
      }
   }
   // printf("main color %d\n", GetMainColor());
   UChar_t c[4] = {0, 0, 0, 0};
   REveUtil::ColorFromIdx(GetMainColor(), c);
   // printf("rgb %d %d %d\n", c[0], c[1], c[2]);
   return (c[0] << 16) + (c[1] << 8) + c[2]; // AMT perhaps this can be ignored
}

float REveBoxSet::GetColorFromDigitAsFloat(REveDigitSet::DigitBase_t &digit)
{
   uint32_t c = GetColorFromDigit(digit);
   // this line required to avoid strict-aliasing rules warning
   auto pc = (float *) &c;
   return *pc;
}


////////////////////////////////////////////////////////////////////////////////
/// Write shape data for different cases
///
void REveBoxSet::WriteShapeData(REveDigitSet::DigitBase_t &digit)
{
   switch (fBoxType) {
   case REveBoxSet::kBT_Vertices: {
      REveBoxSet::Vertices_t &b = (REveBoxSet::Vertices_t &)(digit);
      // vertices
      for (int c = 0; c < 8; c++) {
         for (int j = 0; j < 3; j++)
            fRenderData->PushV(b.fVertices[c][j]);
      }
      break;
   }

   case REveBoxSet::kBT_InstancedScaled: {
      InstancedScaled_t &b = (InstancedScaled_t &)(digit);
      // position
      fRenderData->PushV(b.fX, b.fY, b.fZ);
      fRenderData->PushV(GetColorFromDigitAsFloat(b)); // color ?
      fRenderData->PushV(b.fW, b.fH, b.fD);
      fRenderData->PushV(2.f); // trasp ?
      break;
   }
   case REveBoxSet::kBT_Instanced: {
      Instanced_t &b =(Instanced_t &)(digit);
      // position
      fRenderData->PushV(b.fX, b.fY, b.fZ);
      fRenderData->PushV(GetColorFromDigitAsFloat(b)); // color ?
      fRenderData->PushV(2.f); // trasp ?
      break;
   }
   case REveBoxSet::kBT_InstancedScaledRotated: {
      InstancedScaledRotated_t &b = (InstancedScaledRotated_t &)(digit);
      float* a = b.fMat;
      fRenderData->PushV(a[12], a[13], a[14]);
      fRenderData->PushV(GetColorFromDigitAsFloat(b));
      // write the first three columns
      for (int i = 0; i < 12; i++) {
         fRenderData->PushV(a[i]);
      }
      break;
   }

   default: assert(false && "REveBoxSet::BuildRenderData only kBT_Vertices type supported");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill the structure with a random set of boxes.

void REveBoxSet::Test(Int_t nboxes)
{
   Reset(kBT_InstancedScaled, kTRUE, nboxes);
   TRandom rnd(0);
   const Float_t origin = 10, size = 2;
   Int_t color;
   for(Int_t i=0; i<nboxes; ++i)
   {
      AddInstanceScaled(origin * rnd.Uniform(-1, 1),
             origin * rnd.Uniform(-1, 1),
             origin * rnd.Uniform(-1, 1),
             size   * rnd.Uniform(0.1, 1),
             size   * rnd.Uniform(0.1, 1),
             size   * rnd.Uniform(0.1, 1));

      REveUtil::ColorFromIdx(rnd.Integer(256), (UChar_t*)&color);
      DigitValue(color);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Use instancing in RenderCore.

bool REveBoxSet::Instanced()
{
   return gEve->IsRCore() && (fBoxType != kBT_Vertices);
}


/*
////////////////////////////////////////////////////////////////////////////////
/// Set DigitShape

bool REveBoxSet::SetDigitShape(const std::vector<float>& vrtBuff, const std::vector<int>& idxBuff)
{
   fGeoShape.v = v;

}
*/
