// @(#)root/graf:$Name$:$Id$
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

#ifndef __CINT__
#include <fstream.h>
#else
class ofstream;
#endif

#ifndef ROOT_TText
#include "TText.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLatex                                                               //
//                                                                      //
// The Latex-style text processor class                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


struct FormSize_t {
      Float_t width,dessus,dessous;
};

struct TextSpec_t {
   Float_t angle,size;
   Int_t color,font;
};

// compute size of a portion of a formula
class FormSize {
private:
      Float_t width, dessus, dessous;
public:
      FormSize() { width=0; dessus=0; dessous=0; } // constructeur par defaut
      FormSize(Float_t x, Float_t y1, Float_t y2) { width=x; dessus=y1; dessous=y2; } // constructeur
      virtual ~FormSize() {} //destructeur

      // definition de l'operateur + et +=
      FormSize operator+(FormSize f)
         { return FormSize(f.Width()+width,TMath::Max(f.Dessus(),dessus),TMath::Max(f.Dessous(),dessous)); }
      void operator+=(FormSize f)
         { width += f.Width(); dessus = TMath::Max(dessus,f.Dessus()); dessous = TMath::Max(dessous,f.Dessous()); }

      FormSize add_Dessus(FormSize f)
         { return FormSize(f.Width()+width,f.Height()+dessus,dessous); }
      FormSize add_Dessous(FormSize f)
         { return FormSize(f.Width()+width,dessus,f.Height()+dessous); }
      FormSize add_DessusDessous(FormSize f1, FormSize f2)
         { return FormSize(width+TMath::Max(f1.Width(),f2.Width()),dessus+f1.Dessus(),dessous+f2.Dessous()); }

      // retour des valeurs
      inline Float_t Width()   { return width; }
      inline Float_t Dessus()  { return dessus; }
      inline Float_t Dessous() { return dessous; }
      inline Float_t Height()  { return dessus+dessous; }
};

class TLatex : public TText, public TAttLine {
protected:
      Float_t       fFactorSize;      // Relative size of subscripts and superscripts
      Float_t       fFactorPos;       // Relative position of subscripts and superscripts
      Int_t         fLimitFactorSize; // lower bound for subscripts/superscripts size
      const Char_t *fError;           //!error code
      Bool_t        fShow;            // is true during the second pass (Painting)
      FormSize_t   *fTabSize;         //!array of values for the different zones
      Float_t       fOriginSize;      // Font size of the starting font
      Int_t         fTabMax;          // Maximum allocation for array fTabSize;
      Int_t         fPos;             // Current position in array fTabSize;

      //Text analysis and painting
      FormSize Analyse(Float_t x, Float_t y, TextSpec_t spec, const Char_t* t,Int_t length);
      FormSize Anal1(TextSpec_t spec, const Char_t* t,Int_t length);

      void DrawLine(Float_t x1, Float_t y1, Float_t x2, Float_t y2, TextSpec_t spec);
      void DrawParenthesis(Float_t x1, Float_t y1, Float_t r1, Float_t r2, Float_t phimin, Float_t phimax, TextSpec_t spec);

      FormSize FirstParse(Float_t angle, Float_t size, const Char_t *text);

      void Savefs(FormSize *fs);
      FormSize Readfs();

      Int_t CheckLatexSyntax(TString &text) ;

public:
      // TLatex status bits
      enum { kTextNDC = BIT(14) };

      TLatex();
      TLatex(Coord_t x, Coord_t y, const char *text);
      TLatex(const TLatex &text);
      virtual ~TLatex();
      void     Copy(TObject &text);

      TLatex  *DrawLatex(Float_t x, Float_t y, const char *text);
      Float_t  GetHeight();
      void     GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
      Float_t  GetXsize();
      Float_t  GetYsize();
      void     Paint(Option_t *option="");
      void     PaintLatex(Float_t x, Float_t y, Float_t angle, Float_t size, const char *text);

      void     SavePrimitive(ofstream &out, Option_t *option);
      void     SetIndiceSize(Float_t factorSize);
      void     SetLimitIndiceSize(Int_t limitFactorSize);

      ClassDef(TLatex,1)  //The Latex-style text processor class
};

#endif
