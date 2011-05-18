// @(#)root/gpad:$Id$
// Author: Rene Brun   01/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TButton
#define ROOT_TButton


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TButton                                                              //
//                                                                      //
//  A TButton object is a specialized TPad including possible list
//  of primitives used to build selections and options menus in a canvas.
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPad
#include "TPad.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

class TButton : public TPad, public TAttText {

private:
   Bool_t fFocused;     // If cursor is in...
   Bool_t fFraming;     // True if you want a frame to be painted when pressed

   TButton(const TButton &org);            // no copy ctor, use TObject::Clone()
   TButton &operator=(const TButton &rhs); // idem

protected:
   TString      fMethod;      //Method to be executed by this button

public:
   TButton();
   TButton(const char *title, const char *method, Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   virtual ~TButton();
   virtual void  Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0);
   virtual void  Draw(Option_t *option="");
   virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual const char *GetMethod() const { return fMethod.Data(); }
   virtual void  Paint(Option_t *option="");
   virtual void  PaintModified();
   virtual void  Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   virtual void  SavePrimitive(ostream &out, Option_t *option = "");
   virtual void  SetBorderMode(Short_t bordermode) { fBorderMode = bordermode; }
   virtual void  SetFraming(Bool_t f=1);
   virtual Bool_t GetFraming() { return fFraming; };
   virtual void  SetGrid(Int_t valuex = 1, Int_t valuey = 1);
   virtual void  SetLogx(Int_t value = 1);
   virtual void  SetLogy(Int_t value = 1);
   virtual void  SetMethod(const char *method) { fMethod=method; } // *MENU*
   virtual void  SetName(const char *name) { fName = name; }
   virtual void  x3d(Option_t *option="");

   ClassDef(TButton,0)  //A user interface button.
};

inline void TButton::Divide(Int_t, Int_t, Float_t, Float_t, Int_t) { }
inline void TButton::SetGrid(Int_t, Int_t) { }
inline void TButton::SetLogx(Int_t) { }
inline void TButton::SetLogy(Int_t) { }
inline void TButton::x3d(Option_t *) { }

#endif

