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
      kBT_Undef,                  // unknown-ignored
      kBT_Vertices,               // arbitrary box: specify its 8 vertices, 8*(x,y,z)
      kBT_FreeBox = kBT_Vertices, // the old name of kBT_Vertices
      kBT_Instanced,              // axis-aligned box of the default size, see SetDefWidth(): specify its centre
      kBT_InstancedScaled,        // axis-aligned box: specify its centre and half size
      kBT_InstancedScaledRotated  // any transformation of the instance shape: specify its matrix
   };

   /// The shape every instance of a kBT_Instanced* box set draws.
   enum EInstanceShape_e {
      kIS_Box,
      kIS_Hex,
      kIS_Cone,
      kIS_ConeCapped
   };

   struct Vertices_t : public DigitBase_t {
      Float_t fVertices[8][3];
   }; // the 8 vertices
   struct Instanced_t : public DigitBase_t {
      Float_t fX, fY, fZ;
   }; // centre
   struct InstancedScaled_t : public Instanced_t {
      Float_t fW, fH, fD;
   }; // centre and half size
   struct InstancedScaledRotated_t : public DigitBase_t {
      Float_t fMat[16];
   }; // column-major matrix

   using BFreeBox_t = Vertices_t; // the old name of Vertices_t

protected:
   EBoxType_e fBoxType;                      // Type of rendered box.
   EInstanceShape_e fInstanceShape{kIS_Box}; // Shape of the instances of the kBT_Instanced* types.

   Float_t fDefHalfWidth{0.5};  // Half size along x of a kBT_Instanced box.
   Float_t fDefHalfHeight{0.5}; // Half size along y of a kBT_Instanced box.
   Float_t fDefHalfDepth{0.5};  // Half size along z of a kBT_Instanced box.

   Int_t fBoxSkip; // Number of boxes to skip for each drawn box during scene rotation.

   Bool_t fDrawConeCap{false}; // Close the base of a cone.

   int fTexX{0}, fTexY{0}; // Size of the instance-data texture.

   static Int_t SizeofAtom(EBoxType_e bt);
   void InstanceHalfSize(const DigitBase_t &digit, Float_t half[3]) const;
   void WriteShapeData(REveDigitSet::DigitBase_t &digit);
   unsigned int GetColorFromDigit(REveDigitSet::DigitBase_t &digit);
   float GetColorFromDigitAsFloat(REveDigitSet::DigitBase_t &digit);

public:
   REveBoxSet(const char* n="REveBoxSet", const char* t="");
   ~REveBoxSet() override {}

   void Reset(EBoxType_e boxType, Bool_t valIsCol, Int_t chunkSize);
   void Reset();

   void AddVertices(const Float_t *verts);
   void AddFreeBox(const Float_t *verts) { AddVertices(verts); } // the old name of AddVertices()
   void AddBox(const Float_t *verts) { AddVertices(verts); }     // the TEveBoxSet name of AddVertices()
   void AddInstanceScaled(Float_t cx, Float_t cy, Float_t cz, Float_t hw, Float_t hh, Float_t hd);
   void AddInstanceScaledCorner(Float_t x, Float_t y, Float_t z, Float_t w, Float_t h, Float_t d);
   void AddInstance(Float_t cx, Float_t cy, Float_t cz);
   void AddInstanceCorner(Float_t x, Float_t y, Float_t z);
   void AddInstanceMat4(const Float_t* mat4);

   void AddCone(const REveVector& pos, const REveVector& dir, Float_t r);
   void AddEllipticCone(const REveVector& pos, const REveVector& dir, Float_t r, Float_t r2, Float_t angle=0);

   void AddHex(const REveVector& pos, Float_t r, Float_t angle, Float_t depth);

   EInstanceShape_e GetInstanceShape() const { return fInstanceShape; }
   void SetInstanceShape(EInstanceShape_e s) { fInstanceShape = s; }

   void ComputeBBox() override;

   void Test(Int_t nboxes);

   // The size of a kBT_Instanced box, full or half.
   Float_t GetDefWidth() const { return 2 * fDefHalfWidth; }
   Float_t GetDefHeight() const { return 2 * fDefHalfHeight; }
   Float_t GetDefDepth() const { return 2 * fDefHalfDepth; }
   Float_t GetDefHalfWidth() const { return fDefHalfWidth; }
   Float_t GetDefHalfHeight() const { return fDefHalfHeight; }
   Float_t GetDefHalfDepth() const { return fDefHalfDepth; }

   void SetDefWidth(Float_t v) { fDefHalfWidth = 0.5f * v; }
   void SetDefHeight(Float_t v) { fDefHalfHeight = 0.5f * v; }
   void SetDefDepth(Float_t v) { fDefHalfDepth = 0.5f * v; }
   void SetDefHalfWidth(Float_t v) { fDefHalfWidth = v; }
   void SetDefHalfHeight(Float_t v) { fDefHalfHeight = v; }
   void SetDefHalfDepth(Float_t v) { fDefHalfDepth = v; }

   Bool_t GetDrawConeCap() const { return fDrawConeCap; }
   void SetDrawConeCap(Bool_t x)
   {
      fDrawConeCap = x;
      StampObjProps();
   }

   Int_t GetBoxSkip()   const { return fBoxSkip; }
   void  SetBoxSkip(Int_t bs) { fBoxSkip = bs; }

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void  BuildRenderData() override;

   bool Instanced();
};

} // namespace Experimental
} // namespace ROOT
#endif
