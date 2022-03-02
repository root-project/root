// @(#)root/histpainter:$Id$
// Author: Rene Brun   15/11/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPaletteAxis
#define ROOT_TPaletteAxis


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPaletteAxis                                                         //
//                                                                      //
// class used to display a color palette axis for 2-d plots             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPave.h"
#include "TGaxis.h"

class TH1;

class TPaletteAxis : public TPave {

protected:
   TGaxis       fAxis;          ///<  Palette axis
   TH1         *fH;             ///<! Pointer to parent histogram

public:
   // TPaletteAxis status bits
   enum EStatusBits { kHasView = BIT(11) };

   TPaletteAxis();
   TPaletteAxis(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, TH1 *h);
   TPaletteAxis(const TPaletteAxis &palette);
   ~TPaletteAxis() override;
   void Copy(TObject &palette) const override;
   TPaletteAxis& operator=(const TPaletteAxis&);

   Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
   void  ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   TGaxis       *GetAxis() {return &fAxis;}
   Int_t         GetBinColor(Int_t i, Int_t j);
   TH1*          GetHistogram(){return fH;}
   char *GetObjectInfo(Int_t px, Int_t py) const override;
   Int_t         GetValueColor(Double_t zc);
   void  Paint(Option_t *option="") override;
   void  SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void          SetHistogram(TH1* h) {fH = h;}
   virtual void  SetLabelColor(Int_t labelcolor) {fAxis.SetLabelColor(labelcolor);} // *MENU*
   virtual void  SetLabelFont(Int_t labelfont) {fAxis.SetLabelFont(labelfont);} // *MENU*
   virtual void  SetLabelOffset(Float_t labeloffset) {fAxis.SetLabelOffset(labeloffset);} // *MENU*
   virtual void  SetLabelSize(Float_t labelsize) {fAxis.SetLabelSize(labelsize);} // *MENU*
   virtual void  SetTitleOffset(Float_t titleoffset=1) {fAxis.SetTitleOffset(titleoffset);} // *MENU*
   virtual void  SetTitleSize(Float_t titlesize) {fAxis.SetTitleSize(titlesize);} // *MENU*
   void  SetLineColor(Color_t linecolor) override {fAxis.SetLineColor(linecolor);} // *MENU*
   void  SetLineWidth(Width_t linewidth) override {fAxis.SetLineWidth(linewidth);} // *MENU*
   virtual void  UnZoom();  // *MENU*

   ClassDefOverride(TPaletteAxis,4)  //class used to display a color palette axis for 2-d plots
};

#endif

