// @(#)root/eve7:$Id$
// Author: Matevz Tadel, 2010, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveShape
#define ROOT7_REveShape

#include <ROOT/REveElement.hxx>
#include <ROOT/REveVector.hxx>

#include "TAttBBox.h"
#include "TColor.h"

namespace ROOT {
namespace Experimental {

// =========================================================================
// REveShape
// Abstract base-class for 2D/3D shapes.
// =========================================================================

class REveShape : public REveElement,
                  public TAttBBox
{
private:
   REveShape(const REveShape &) = delete;
   REveShape &operator=(const REveShape &) = delete;

public:
   typedef std::vector<REveVector2> vVector2_t;

protected:
   Color_t fFillColor = 7;   // fill color of polygons
   Color_t fLineColor = 12;  // outline color of polygons
   UChar_t fFillAlpha = 255; // alpha of the fill
   UChar_t fLineAlpha = 255; // alpha of outline
   Float_t fLineWidth = 1;   // outline width of polygons

   Bool_t fDrawFrame      = true;  // draw frame
   Bool_t fHighlightFrame = false; // highlight frame / all shape
   Bool_t fMiniFrame      = true;  // draw minimal frame

public:
   REveShape(const std::string &n = "REveShape", const std::string &t = "");
   ~REveShape() override;

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;

   // Rendering parameters.
   void SetMainColor(Color_t color) override;

   Color_t GetFillColor() const { return fFillColor; }
   Color_t GetLineColor() const { return fLineColor; }
   UChar_t GetFillAlpha() const { return fFillAlpha; }
   UChar_t GetLineAlpha() const { return fLineAlpha; }
   Float_t GetLineWidth() const { return fLineWidth; }
   Bool_t GetDrawFrame() const { return fDrawFrame; }
   Bool_t GetHighlightFrame() const { return fHighlightFrame; }
   Bool_t GetMiniFrame() const { return fMiniFrame; }

   void SetFillColor(Color_t c) { fFillColor = c; StampObjProps(); }
   void SetLineColor(Color_t c) { fLineColor = c; StampObjProps(); }
   void SetFillAlpha(UChar_t c) { fFillAlpha = c; StampObjProps(); }
   void SetLineAlpha(UChar_t c) { fLineAlpha = c; StampObjProps(); }
   void SetLineWidth(Float_t lw) { fLineWidth = lw; StampObjProps(); }
   void SetDrawFrame(Bool_t f) { fDrawFrame = f; StampObjProps(); }
   void SetHighlightFrame(Bool_t f) { fHighlightFrame = f; StampObjProps(); }
   void SetMiniFrame(Bool_t r) { fMiniFrame = r; StampObjProps(); }

   // ----------------------------------------------------------------

   void CopyVizParams(const REveElement *el) override;
   void WriteVizParams(std::ostream &out, const TString &var) override;

   // ----------------------------------------------------------------

   // Abstract function from TAttBBox:
   // virtual void ComputeBBox();

   // ----------------------------------------------------------------

   static Int_t FindConvexHull(const vVector2_t &pin, vVector2_t &pout, REveElement *caller = nullptr);

   static Bool_t IsBoxOrientationConsistentEv(const REveVector box[8]);
   static Bool_t IsBoxOrientationConsistentFv(const Float_t box[8][3]);

   static void CheckAndFixBoxOrientationEv(REveVector box[8]);
   static void CheckAndFixBoxOrientationFv(Float_t box[8][3]);
};

} // namespace Experimental
} // namespace ROOT

#endif
