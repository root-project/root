// @(#)root/eve7:$Id$
// Author: Matevz Tadel

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveSMorph
#define ROOT7_REveSMorph

#include <ROOT/REveElement.hxx>

#include <TAttBBox.h>

#include <algorithm>
#include <string>

namespace ROOT {
namespace Experimental {


class REveSMorph : public REveElement,
                   public TAttBBox
{
private:
   REveSMorph(const REveSMorph &) = delete;
   REveSMorph &operator=(const REveSMorph &) = delete;

protected:
   // Tessellation. GUI range [2, 100] / [3, 100], step 1.
   Int_t   fTLevel{24};     ///< divisions in theta
   Int_t   fPLevel{32};     ///< divisions in phi

   // Morph. All zero is a plain sphere. GUI range [-2, 2], step 0.01; the
   // shape stops being informative well before the ends.
   Float_t fTx{0.f};        ///< twist of phi, proportional to cos(theta)
   Float_t fCx{0.f};        ///< radial convergence, proportional to cos(theta)
   Float_t fRz{0.f};        ///< shear about z, proportional to x

   // Extent, as fractions of the full ranges. GUI range [0, 1], step 0.001.
   Float_t fThetaMin{0.f};  ///< theta from fThetaMin * pi
   Float_t fThetaMax{1.f};  ///< theta to   fThetaMax * pi
   Float_t fPhiMean{0.5f};  ///< phi centred on fPhiMean * 2pi
   Float_t fPhiRange{1.f};  ///< phi spanning fPhiRange * 2pi; 1 closes the seam

   /// Space the theta rings by equal surface area rather than by equal angle,
   /// which keeps the quads near the poles from collapsing.
   Bool_t  fEquiSurf{kFALSE};

   /// Surface colour, multiplied by the texture where there is one. Must be a
   /// member with SetMainColorPtr() pointing at it, or SetMainColor() is a
   /// silent no-op.
   Color_t fColor{kWhite};

   /// File name under ui5/eve7/textures/. Empty draws in the main colour.
   std::string fTexture;

   Float_t fTexX0{0.f};     ///< u offset
   Float_t fTexY0{1.f};     ///< v offset
   Float_t fTexXC{1.f};     ///< u wraps per turn in phi
   Float_t fTexYC{-1.f};    ///< v wraps per sweep in theta
   Float_t fTexYOff{0.f};   ///< u shift per whole v, for a brick bond

public:
   REveSMorph(const std::string &n = "REveSMorph", const std::string &t = "");
   ~REveSMorph() override = default;

   /// Shape and texture parameters; each setter clamps and stamps kCBObjProps.
   /// Size is in the transformation, see SetRadius().
   /// @{
   Int_t GetTLevel() const { return fTLevel; }
   Int_t GetPLevel() const { return fPLevel; }
   void  SetTLevel(Int_t l) { fTLevel = std::clamp(l, 2, 200); StampObjProps(); }
   void  SetPLevel(Int_t l) { fPLevel = std::clamp(l, 3, 200); StampObjProps(); }

   Float_t GetTx() const { return fTx; }
   Float_t GetCx() const { return fCx; }
   Float_t GetRz() const { return fRz; }
   void    SetTx(Float_t v) { fTx = std::clamp(v, -2.f, 2.f); StampObjProps(); }
   void    SetCx(Float_t v) { fCx = std::clamp(v, -2.f, 2.f); StampObjProps(); }
   void    SetRz(Float_t v) { fRz = std::clamp(v, -2.f, 2.f); StampObjProps(); }

   Float_t GetThetaMin()  const { return fThetaMin; }
   Float_t GetThetaMax()  const { return fThetaMax; }
   Float_t GetPhiMean()   const { return fPhiMean; }
   Float_t GetPhiRange()  const { return fPhiRange; }
   void    SetThetaMin(Float_t v)  { fThetaMin = std::clamp(v, 0.f, 1.f); StampObjProps(); }
   void    SetThetaMax(Float_t v)  { fThetaMax = std::clamp(v, 0.f, 1.f); StampObjProps(); }
   void    SetPhiMean(Float_t v)   { fPhiMean  = std::clamp(v, 0.f, 1.f); StampObjProps(); }
   void    SetPhiRange(Float_t v)  { fPhiRange = std::clamp(v, 0.f, 1.f); StampObjProps(); }

   Bool_t GetEquiSurf() const { return fEquiSurf; }
   void   SetEquiSurf(Bool_t x) { fEquiSurf = x; StampObjProps(); }

   const std::string &GetTexture() const { return fTexture; }
   void SetTexture(const std::string &f) { fTexture = f; StampObjProps(); }

   Float_t GetTexX0()   const { return fTexX0; }
   Float_t GetTexY0()   const { return fTexY0; }
   Float_t GetTexXC()   const { return fTexXC; }
   Float_t GetTexYC()   const { return fTexYC; }
   Float_t GetTexYOff() const { return fTexYOff; }
   void    SetTexX0(Float_t v)   { fTexX0   = std::clamp(v, -1e3f, 1e3f); StampObjProps(); }
   void    SetTexY0(Float_t v)   { fTexY0   = std::clamp(v, -1e3f, 1e3f); StampObjProps(); }
   void    SetTexXC(Float_t v)   { fTexXC   = std::clamp(v, -1e3f, 1e3f); StampObjProps(); }
   void    SetTexYC(Float_t v)   { fTexYC   = std::clamp(v, -1e3f, 1e3f); StampObjProps(); }
   void    SetTexYOff(Float_t v) { fTexYOff = std::clamp(v, -1e3f, 1e3f); StampObjProps(); }
   /// @}

   void SetRadius(Float_t r);

   void ComputeBBox() override;

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void  BuildRenderData() override;
};

} // namespace Experimental
} // namespace ROOT

#endif
