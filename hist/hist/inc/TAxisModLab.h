// @(#)root/graf:$Id$
// Author: Olivier Couet

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

   Int_t   fLabNum;   ///< Label number.
   TString fLabText;  ///< Label text

public:

   TAxisModLab();

   void SetLabNum(Int_t n = 0);
   void SetAngle(Double_t a = -1.);
   void SetSize(Double_t s = -1.);
   void SetAlign(Int_t a = -1);
   void SetColor(Int_t c = -1);
   void SetFont(Int_t f = -1);
   void SetText(TString t = "");

   Int_t    GetLabNum() {return fLabNum;}
   Double_t GetAngle()  {return fTextAngle;}
   Double_t GetSize()   {return fTextSize;}
   Int_t    GetAlign()  {return fTextAlign;}
   Int_t    GetColor()  {return fTextColor;}
   Int_t    GetFont()   {return fTextFont;}
   TString  GetText()   {return fLabText;}

   ClassDefOverride(TAxisModLab,3)  // Modified axis label
};

#endif
