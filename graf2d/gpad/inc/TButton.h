// @(#)root/gpad:$Id$
// Author: Rene Brun   01/07/96

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TButton
#define ROOT_TButton

#include "TPad.h"
#include "TAttText.h"

class TButton : public TPad, public TAttText {

private:
   Bool_t fFocused;     ///< If cursor is in...
   Bool_t fFraming;     ///< True if you want a frame to be painted when pressed
   UChar_t fValidPattern[128];  ///<! pattern in memory to detect button deletion

   TButton(const TButton &) = delete;
   TButton &operator=(const TButton &) = delete;

protected:
   TString      fMethod;      ///< Method to be executed by this button

public:
   TButton();
   TButton(const char *title, const char *method, Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   ~TButton() override;
   void Divide(Int_t = 1, Int_t = 1, Float_t = 0.01, Float_t = 0.01, Int_t = 0) override {}
   void Draw(Option_t *option = "") override;
   void ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual const char *GetMethod() const { return fMethod.Data(); }
   void Paint(Option_t *option = "") override;
   void PaintModified() override;
   void Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2) override;
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void SetBorderMode(Short_t bordermode) override { fBorderMode = bordermode; }
   virtual void SetFraming(Bool_t f = kTRUE);
   virtual Bool_t GetFraming() { return fFraming; };
   void SetGrid(Int_t = 1, Int_t = 1) override {}
   void SetLogx(Int_t = 1) override {}
   void SetLogy(Int_t = 1) override {}
   virtual void SetMethod(const char *method) { fMethod = method; } // *MENU*
   void SetName(const char *name) override { fName = name; }
   void x3d(Option_t * /*option */ = "") override {}

   ClassDefOverride(TButton,0)  //A user interface button.
};

#endif

