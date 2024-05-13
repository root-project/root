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
#include "TH1.h"

class TPaletteAxis : public TPave {

protected:
   TGaxis       fAxis;          ///<  Palette axis
   TH1         *fH;             ///<! Pointer to parent histogram

public:
   // TPaletteAxis status bits
   enum EStatusBits { kHasView = BIT(11) };

   TPaletteAxis();
   TPaletteAxis(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, TH1 *h);
   TPaletteAxis(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, Double_t min, Double_t max);
   TPaletteAxis(Double_t x1, Double_t y1,Double_t x2 ,Double_t y2, TAxis *ax);
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
   virtual void  SetNdivisions(Int_t ndiv=10)                 {if (fH) fH->GetZaxis()->SetNdivisions(ndiv);       else fAxis.SetNdivisions(ndiv);}     // *MENU*
   virtual void  SetLabelColor(Int_t color=1)                 {if (fH) fH->GetZaxis()->SetLabelColor(color);      else fAxis.SetLabelColor(color);}    // *MENU*
   virtual void  SetLabelFont(Int_t font=42)                  {if (fH) fH->GetZaxis()->SetLabelFont(font);        else fAxis.SetLabelFont(font);}      // *MENU*
   virtual void  SetLabelOffset(Float_t offset=0.005)         {if (fH) fH->GetZaxis()->SetLabelOffset(offset);    else fAxis.SetLabelOffset(offset);}  // *MENU*
   virtual void  SetLabelSize(Float_t size=0.035)             {if (fH) fH->GetZaxis()->SetLabelSize(size);        else fAxis.SetLabelSize(size);}      // *MENU*
   virtual void  SetMaxDigits(Float_t maxdigits=5)            {if (fH) fH->GetZaxis()->SetMaxDigits(maxdigits);   else fAxis.SetMaxDigits(maxdigits);} // *MENU*
   virtual void  SetTickLength(Float_t length=0.03)           {if (fH) fH->GetZaxis()->SetTickLength(length);     else fAxis.SetTickLength(length);}   // *MENU*
   virtual void  SetTitleOffset(Float_t offset=1)             {if (fH) fH->GetZaxis()->SetTitleOffset(offset);    else fAxis.SetTitleOffset(offset);}  // *MENU*
   virtual void  SetTitleSize(Float_t size=0.035)             {if (fH) fH->GetZaxis()->SetTitleSize(size);        else fAxis.SetTitleSize(size);}      // *MENU*
   virtual void  SetTitleColor(Int_t color=1)                 {if (fH) fH->GetZaxis()->SetTitleColor(color);      else fAxis.SetTitleColor(color);}    // *MENU*
   virtual void  SetTitleFont(Int_t font=42)                  {if (fH) fH->GetZaxis()->SetTitleFont(font);        else fAxis.SetTitleFont(font);}      // *MENU*
   virtual void  SetTitle(const char *title="")               {if (fH) fH->GetZaxis()->SetTitle(title);           else fAxis.SetTitle(title);}         // *MENU*
   virtual void  SetAxisColor(Int_t color=1, Float_t alpha=1) {if (fH) fH->GetZaxis()->SetAxisColor(color,alpha);} // *MENU*
   void  SetLineWidth(Width_t width) override {fAxis.SetLineWidth(width);} // *MENU*

   virtual void  UnZoom();  // *MENU*

   ClassDefOverride(TPaletteAxis,4)  //class used to display a color palette axis for 2-d plots
};

#endif

