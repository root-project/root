// @(#)root/graf:$Id: TMathText.h 20882 2007-11-19 11:31:26Z rdm $
// Author: Yue Shi Lai   12/12/09

/*************************************************************************
 * Copyright (C) 2009, Yue Shi Lai.                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TMathText
#define ROOT_TMathText

#include "TText.h"
#include "TAttFill.h"

class TMathTextRenderer;

class TMathText : public TText, public TAttFill {
protected:
      void *fRenderer; //!TMathText Painter
      TMathText &operator=(const TMathText &);

      void Render(const Double_t x, const Double_t y,
               const Double_t size, const Double_t angle,
               const Char_t *t, const Int_t length);
      void GetSize(Double_t &x0, Double_t &y0,
                Double_t &x1, Double_t &y1,
                const Double_t size, const Double_t angle,
                const Char_t *t, const Int_t length);
      void GetAlignPoint(Double_t &x0, Double_t &y0,
                     const Double_t size, const Double_t angle,
                     const Char_t *t, const Int_t length,
                     const Short_t align);
public:
      enum {
         kTextNDC = BIT(14)
      };
      TMathText();
      TMathText(Double_t x, Double_t y, const char *text);
      TMathText(const TMathText &text);
      virtual ~TMathText();
      void Copy(TObject &text) const override;
      TMathText *DrawMathText(Double_t x, Double_t y, const char *text);
      void GetBoundingBox(UInt_t &w, UInt_t &h, Bool_t angle = kFALSE) override;
      Double_t GetXsize();
      Double_t GetYsize();
      void Paint(Option_t *option = "") override;
      virtual void PaintMathText(Double_t x, Double_t y, Double_t angle, Double_t size, const char *text);
      void SavePrimitive(std::ostream &out, Option_t *option = "") override;
      friend class TMathTextRenderer;

      ClassDefOverride(TMathText,2) //TeX mathematical formula
};

#endif
