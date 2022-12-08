// @(#)root/graf:$Id$
// Author: Nicolas Brun   07/08/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TLatex
#define ROOT_TLatex

#include "TText.h"
#include "TAttLine.h"
#include <vector>

class TLatex : public TText, public TAttLine {
protected:

////////////////////////////////////////////////////////////////////////////////
///  @brief TLatex helper struct holding the attributes of a piece of text.

   struct TextSpec_t {
      Double_t fAngle,fSize;
      Int_t    fColor,fFont;
   };

////////////////////////////////////////////////////////////////////////////////
/// @class TLatexFormSize
/// @brief TLatex helper class used to compute the size of a portion of a formula.

   class TLatexFormSize {
   private:
      Double_t fWidth{0}, fOver{0}, fUnder{0};

   public:
      TLatexFormSize() = default;
      TLatexFormSize(Double_t width, Double_t over, Double_t under) : fWidth(width), fOver(over), fUnder(under) { } // constructor

      // definition of operators + and +=
      TLatexFormSize operator+(TLatexFormSize f)
      { return TLatexFormSize(f.Width()+fWidth,TMath::Max(f.Over(),fOver),TMath::Max(f.Under(),fUnder)); }
      void operator+=(TLatexFormSize f)
      { fWidth += f.Width(); fOver = TMath::Max(fOver,f.Over()); fUnder = TMath::Max(fUnder,f.Under()); }

      inline void Set(Double_t x, Double_t y1, Double_t y2) { fWidth=x; fOver=y1; fUnder=y2; }
      TLatexFormSize AddOver(TLatexFormSize f)
      { return TLatexFormSize(f.Width()+fWidth,f.Height()+fOver,fUnder); }
      TLatexFormSize AddUnder(TLatexFormSize f)
      { return TLatexFormSize(f.Width()+fWidth,fOver,f.Height()+fUnder); }
      TLatexFormSize AddOver(TLatexFormSize f1, TLatexFormSize f2)
      { return TLatexFormSize(fWidth+TMath::Max(f1.Width(),f2.Width()),fOver+f1.Over(),fUnder+f2.Under()); }

      // return members
      inline Double_t Width()  const { return fWidth; }
      inline Double_t Over()   const { return fOver; }
      inline Double_t Under()  const { return fUnder; }
      inline Double_t Height() const { return fOver+fUnder; }
   };

   Double_t                    fFactorSize;      ///<! Relative size of subscripts and superscripts
   Double_t                    fFactorPos;       ///<! Relative position of subscripts and superscripts
   Int_t                       fLimitFactorSize; ///< lower bound for subscripts/superscripts size
   const Char_t               *fError{nullptr};  ///<! error code
   Bool_t                      fShow;            ///<! is true during the second pass (Painting)
   std::vector<TLatexFormSize> fTabSize;         ///<! array of values for the different zones
   Double_t                    fOriginSize;      ///< Font size of the starting font
   Bool_t                      fItalic;          ///<! Currently inside italic operator

   TLatex& operator=(const TLatex&);

   //Text analysis and painting
   TLatexFormSize Analyse(Double_t x, Double_t y, TextSpec_t spec, const Char_t* t,Int_t length);
   TLatexFormSize Anal1(TextSpec_t spec, const Char_t* t,Int_t length);

   void DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, TextSpec_t spec);
   void DrawCircle(Double_t x1, Double_t y1, Double_t r, TextSpec_t spec);
   void DrawParenthesis(Double_t x1, Double_t y1, Double_t r1, Double_t r2, Double_t phimin, Double_t phimax, TextSpec_t spec);

   TLatexFormSize FirstParse(Double_t angle, Double_t size, const Char_t *text);

   Int_t PaintLatex1(Double_t x, Double_t y, Double_t angle, Double_t size, const char *text);

   void Savefs(TLatexFormSize *fs);
   TLatexFormSize Readfs();

   Int_t CheckLatexSyntax(TString &text) ;

public:
   // TLatex status bits
   enum {
      kTextNDC = BIT(14) ///< The text position is in NDC coordinates
   };

   TLatex();
   TLatex(Double_t x, Double_t y, const char *text);
   TLatex(const TLatex &text);
   virtual ~TLatex();

   void             Copy(TObject &text) const override;

   TLatex          *DrawLatex(Double_t x, Double_t y, const char *text);
   TLatex          *DrawLatexNDC(Double_t x, Double_t y, const char *text);

   Double_t         GetHeight() const;
   Double_t         GetXsize();
   Double_t         GetYsize();
   void             GetBoundingBox(UInt_t &w, UInt_t &h, Bool_t angle = kFALSE) override;
   void             Paint(Option_t *option="") override;
   virtual void     PaintLatex(Double_t x, Double_t y, Double_t angle, Double_t size, const char *text);

   void             SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void     SetIndiceSize(Double_t factorSize);
   virtual void     SetLimitIndiceSize(Int_t limitFactorSize);

   ClassDefOverride(TLatex,2)  //The Latex-style text processor class
};

#endif
