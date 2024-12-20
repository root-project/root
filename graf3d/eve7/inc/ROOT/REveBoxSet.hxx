// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveBoxSet
#define ROOT_REveBoxSet

#include "ROOT/REveDigitSet.hxx"
#include "ROOT/REveVector.hxx"

class TGeoMatrix;
class TRandom;

namespace ROOT {
namespace Experimental {
class REveBoxSet: public REveDigitSet
{
   friend class REveBoxSetGL;

   REveBoxSet(const REveBoxSet&) = delete;
   REveBoxSet& operator=(const REveBoxSet&) = delete;

public:
   enum EBoxType_e {
      kBT_Undef,           // unknown-ignored
      kBT_FreeBox,         // arbitrary box: specify 8*(x,y,z) box corners
      kBT_Instanced,              // axis-aligned digit w/ fixed dimensions: specify (x,y,z)
      kBT_InstancedScaled,        // axis-aligned digit: specify (x,y,z) and (w, h, d)
      kBT_InstancedScaledRotated  // generic Mat4 transformation
   };

   enum EShape_e {
      kBox,
      kHex,
      kCone,
      kConeCapped
   };

   struct BFreeBox_t       : public DigitBase_t { Float_t fVertices[8][3]; };

   struct Instanced_t      : public DigitBase_t   {  Float_t fX, fY, fZ; }; // save only position == INSTANCED_T

   struct InstancedScaled_t   : public Instanced_t   { Float_t fW, fH, fD; }; // scaled box INSTANCED_SCALED_T

   struct InstancedScaledRotated_t  : public DigitBase_t   { Float_t fMat[16]; }; // INSTANCED_SCALEDROTATED

 // ++ TODO add rotated

protected:
   EBoxType_e        fBoxType;      // Type of rendered box.
   EShape_e          fShapeType{kBox};

   Float_t           fDefWidth  {1};     // Breadth assigned to first coordinate  (A).
   Float_t           fDefHeight {1};    // Breadth assigned to second coordinate (B).
   Float_t           fDefDepth  {1};     // Breadth assigned to third coordinate  (C).

   Int_t             fBoxSkip;      // Number of boxes to skip for each drawn box during scene rotation.

   Bool_t            fDrawConeCap{false};

   int                     fTexX{0}, fTexY{0};
   static Int_t SizeofAtom(EBoxType_e bt);
   void WriteShapeData(REveDigitSet::DigitBase_t &digit);
   unsigned int GetColorFromDigit(REveDigitSet::DigitBase_t &digit);
   float GetColorFromDigitAsFloat(REveDigitSet::DigitBase_t &digit);

public:
   REveBoxSet(const char* n="REveBoxSet", const char* t="");
   ~REveBoxSet() override {}

   void Reset(EBoxType_e boxType, Bool_t valIsCol, Int_t chunkSize);
   void Reset();

   void AddFreeBox(const Float_t* verts);
   void AddInstanceScaled(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h, Float_t d);
   void AddInstance(Float_t a, Float_t b, Float_t c);
   void AddInstanceMat4(const Float_t* mat4);

   void AddBox(const Float_t* verts) { AddFreeBox(verts); }
   void AddBox(Float_t a, Float_t b, Float_t c, Float_t w, Float_t h, Float_t d) { AddInstanceScaled(a, b, c, w, h, d); }
   void AddBox(Float_t a, Float_t b, Float_t c) { AddInstance(a, b, c);}

   void AddCone(const REveVector& pos, const REveVector& dir, Float_t r);
   void AddEllipticCone(const REveVector& pos, const REveVector& dir, Float_t r, Float_t r2, Float_t angle=0);

   void AddHex(const REveVector& pos, Float_t r, Float_t angle, Float_t depth);
   void SetShape(EShape_e x){fShapeType = x;}

   void ComputeBBox() override;

   void Test(Int_t nboxes);

   Float_t GetDefWidth()  const { return fDefWidth;  }
   Float_t GetDefHeight() const { return fDefHeight; }
   Float_t GetDefDepth()  const { return fDefDepth;  }
   Bool_t  GetDrawConeCap() const { return fDrawConeCap;  }

   void SetDefWidth(Float_t v)  { fDefWidth  = v ; }
   void SetDefHeight(Float_t v) { fDefHeight = v ; }
   void SetDefDepth(Float_t v)  { fDefDepth  = v ; }
   void SetDrawConeCap(Bool_t x) { fDrawConeCap=x; StampObjProps(); }

   Int_t GetBoxSkip()   const { return fBoxSkip; }
   void  SetBoxSkip(Int_t bs) { fBoxSkip = bs; }


   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void  BuildRenderData() override;

   bool Instanced();
};

} // namespace Experimental
} // namespace ROOT
#endif
