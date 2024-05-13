// @(#)root/graf:$Id$
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAxisModLab
#define ROOT_TAxisModLab

#include "TObject.h"

#include "TAttText.h"

#include "TString.h"

class TAxisModLab : public TObject, public TAttText {

private:
   Int_t fLabNum;      ///< Label number.
   Double_t fLabValue; ///< Label value, used when label number is 0
   TString fLabText;   ///< Alternative label text

public:

   TAxisModLab();

   void SetLabNum(Int_t n = 0);
   void SetLabValue(Double_t v = 0.);
   void SetAngle(Double_t a = -1.);
   void SetSize(Double_t s = -1.);
   void SetAlign(Int_t a = -1);
   void SetColor(Int_t c = -1);
   void SetFont(Int_t f = -1);
   void SetText(TString t = "");

   Int_t GetLabNum() const { return fLabNum; }
   Double_t GetLabValue() const { return fLabValue; }
   Double_t GetAngle() const { return fTextAngle; }
   Double_t GetSize() const { return fTextSize; }
   Int_t GetAlign() const { return fTextAlign; }
   Int_t GetColor() const { return fTextColor; }
   Int_t GetFont() const { return fTextFont; }
   const TString &GetText() const { return fLabText; }

   ClassDefOverride(TAxisModLab,4)  // Modified axis label
};

#endif
