// @(#)root/eve7:$Id$
// Author: Waad Alshehri, 2023

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveText
#define ROOT7_REveText

#include <ROOT/REveShape.hxx>
#include <ROOT/REveVector.hxx>

namespace ROOT {
namespace Experimental {

//------------------------------------------------------------------------------
// REveText
//------------------------------------------------------------------------------

class REveText : public REveShape
{
private:
   REveText(const REveText &) = delete;
   REveText &operator=(const REveText &) = delete;

protected:
   std::string fText {"<no-text>"};
   /// Default must be a face ROOT ships: only Liberation Mono and Serif are in
   /// $ROOTSYS/fonts, not Sans. Serif is also what the viewer axes hardcode.
   std::string fFont {"LiberationSerif-Regular"};
   REveVector  fPosition {0, 0, 0};
   Float_t     fFontSize {80};
   Float_t     fFontHinting {1.0};
   Float_t     fExtraBorder {0.2}; // border around text in font-size units
   Int_t       fMode {1}; // default mode is in relative screen coordinates [0,1]
   Bool_t      fResizable {true}; // can the client resize it by dragging the corner grip
   Float_t     fFontWeight {0.f}; // synthetic bold; 0 is the font as authored
   Int_t       fAlignH {0};       // EAlignH_e: which point of the box the position refers to
   Int_t       fAlignV {0};       // EAlignV_e
   std::string fClickMir;         // MIR the client sends when this text is clicked; empty = not a button
   ElementId_t fClickTargetId{0}; // element the MIR is addressed to; 0 means this element
   REveAunt   *fClickAunt{nullptr}; //! target held as an aunt, so its death is noticed
   Color_t     fTextColor {kMagenta};
   // UChar_t     fTextAlpha {255}; // Better than main transparency -- to be fixed.

   static std::string sSdfFontDir;

   static bool SetDefaultSdfFontDir();

public:
   /// Anchoring: SetPosition() gives where the text goes, these give which point
   /// of it lands there. kOriginH/kOriginV, the default, is the text origin; the
   /// others are relative to the text box.
   enum EAlignH_e { kOriginH = 0, kLeft, kCenterH, kRight };
   enum EAlignV_e { kOriginV = 0, kTop, kCenterV, kBottom };

   REveText(const Text_t *n = "REveText", const Text_t *t = "");
   virtual ~REveText() {}

    Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void BuildRenderData() override;

   void ComputeBBox() override;

   std::string GetText() const { return fText; }
   void SetText(const std::string &text) { fText = text; StampObjProps(); }

   std::string GetFont() const { return fFont; }
   void SetFont(const std::string &font) { fFont = font; StampObjProps();}

   Float_t GetFontSize() const { return fFontSize; }
   void SetFontSize(float size) { fFontSize = size; StampObjProps();}

   Int_t GetMode() const { return fMode; }
   void SetMode(Int_t mode) { fMode = mode;}

   /// Whether the client may resize this element by dragging its corner grip.
   /// Moving is governed by SetPickable(), so a pickable element that is not
   /// resizable can be repositioned but keeps its size.
   Bool_t GetResizable() const { return fResizable; }
   void SetResizable(Bool_t r) { fResizable = r; StampObjProps(); }

   Int_t GetAlignH() const { return fAlignH; }
   Int_t GetAlignV() const { return fAlignV; }
   void SetTextAlign(Int_t h, Int_t v) { fAlignH = h; fAlignV = v; StampObjProps(); }

   void SetClickAction(const std::string &mir, REveElement *target = nullptr);
   const std::string &GetClickMir() const { return fClickMir; }
   ElementId_t GetClickTargetId() const { return fClickTargetId; }

   void RemoveAunt(REveAunt *au) override;

   /// Stroke weight, applied by moving the SDF threshold rather than loading a
   /// heavier face, so it is continuous and needs no second atlas. Worth a
   /// small positive value for text only a few pixels tall.
   Float_t GetFontWeight() const { return fFontWeight; }
   void SetFontWeight(Float_t w) { fFontWeight = w; StampObjProps(); }

   Float_t GetFontHinting() const { return fFontHinting; }
   void SetFontHinting(Float_t fontHinting) { fFontHinting = fontHinting; StampObjProps();}

   Float_t GetExtraBorder() const { return fExtraBorder; }
   void SetExtraBorder(float size) { fExtraBorder = size; StampObjProps();}

   REveVector GetPosition() const { return fPosition; }
   const REveVector& RefPosition() const { return fPosition; }
   void SetPosition(const REveVector& position) { fPosition = position;}
   void SetPosX(float x) { fPosition.fX = x; StampObjProps(); }
   void SetPosY(float y) { fPosition.fY = y; StampObjProps(); }

   Color_t GetTextColor() const { return fTextColor; }
   void SetTextColor(Color_t color) { fTextColor = color; StampObjProps();}

   // UChar_t GetTextAlpha() const { return fTextAlpha; }
   // void SetTextAlpha(UChar_t c) { fTextAlpha = c; StampObjProps(); }

   static bool SetSdfFontDir(const std::string &dir, bool require_write_access = true);
   static bool AssertSdfFont(const std::string &font_name, const std::string &ttf_font);
};

} // namespace Experimental
} // namespace ROOT

#endif
