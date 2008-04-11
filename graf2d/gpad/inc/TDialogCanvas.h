// @(#)root/gpad:$Id$
// Author: Rene Brun   03/07/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDialogCanvas
#define ROOT_TDialogCanvas


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDialogCanvas                                                        //
//                                                                      //
// A specialized canvas to set attributes.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif
#ifndef ROOT_TAttText
#include "TAttText.h"
#endif

class TDialogCanvas : public TCanvas, public TAttText {

private:
   TDialogCanvas(const TDialogCanvas&);
   TDialogCanvas& operator=(const TDialogCanvas&);

protected:
   TObject     *fRefObject;   //Pointer to object to set attributes
   TPad        *fRefPad;      //Pad containing object

public:
   TDialogCanvas();
   TDialogCanvas(const char *name, const char *title, Int_t ww, Int_t wh);
   TDialogCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, UInt_t ww, UInt_t wh);
   virtual        ~TDialogCanvas();
   virtual void   Apply(const char *action="");
   virtual void   BuildStandardButtons();
   virtual void   Divide(Int_t nx=1, Int_t ny=1, Float_t xmargin=0.01, Float_t ymargin=0.01, Int_t color=0);
   TObject        *GetRefObject() const { return fRefObject; }
   TPad           *GetRefPad() const { return fRefPad; }
   virtual void   Range(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   virtual void   RecursiveRemove(TObject *obj);
   virtual void   SetBorderMode(Short_t bordermode){ fBorderMode = bordermode; }
   virtual void   SetGrid(Int_t valuex = 1, Int_t valuey = 1);
   virtual void   SetLogx(Int_t value = 1);
   virtual void   SetLogy(Int_t value = 1);
   virtual void   SetName(const char *name) { fName = name; }
   virtual void   SetRefObject(TObject*obj) { fRefObject=obj; }
   virtual void   SetRefPad(TPad *pad) { fRefPad=pad; }
   virtual void   x3d(Option_t *option="");

   ClassDef(TDialogCanvas,0)  //A specialized canvas to set attributes.
};

inline void TDialogCanvas::Divide(Int_t, Int_t, Float_t, Float_t, Int_t) { }
inline void TDialogCanvas::SetGrid(Int_t, Int_t) { }
inline void TDialogCanvas::SetLogx(Int_t) { }
inline void TDialogCanvas::SetLogy(Int_t) { }
inline void TDialogCanvas::x3d(Option_t *) { }

#endif

