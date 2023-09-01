// @(#)root/hist:$Id$
// Author: Rene Brun   15/09/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphErrors
#define ROOT_TGraphErrors


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphErrors                                                         //
//                                                                      //
// a Graph with error bars                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph.h"

class TGraphErrors : public TGraph {

protected:
   Double_t    *fEX{nullptr};    ///<[fNpoints] array of X errors
   Double_t    *fEY{nullptr};    ///<[fNpoints] array of Y errors

   void       SwapPoints(Int_t pos1, Int_t pos2) override;

   Double_t** Allocate(Int_t size) override;
   void       CopyAndRelease(Double_t **newarrays,
                             Int_t ibegin, Int_t iend, Int_t obegin) override;
   Bool_t     CopyPoints(Double_t **arrays, Int_t ibegin, Int_t iend,
                         Int_t obegin) override;
   Bool_t     CtorAllocate();
   void       FillZero(Int_t begin, Int_t end,
                       Bool_t from_ctor = kTRUE) override;
   Bool_t     DoMerge(const TGraph * g) override;


public:
   TGraphErrors();
   TGraphErrors(Int_t n);
   TGraphErrors(Int_t n, const Float_t *x, const Float_t *y, const Float_t *ex = nullptr, const Float_t *ey = nullptr);
   TGraphErrors(Int_t n, const Double_t *x, const Double_t *y, const Double_t *ex = nullptr, const Double_t *ey = nullptr);
   TGraphErrors(const TVectorF &vx, const TVectorF &vy, const TVectorF &vex, const TVectorF &vey);
   TGraphErrors(const TVectorD &vx, const TVectorD &vy, const TVectorD &vex, const TVectorD &vey);
   TGraphErrors(const TGraphErrors &gr);
   TGraphErrors& operator=(const TGraphErrors &gr);
   TGraphErrors(const TH1 *h);
   TGraphErrors(const char *filename, const char *format="%lg %lg %lg %lg", Option_t *option="");
   ~TGraphErrors() override;
   void    Apply(TF1 *f) override;
   virtual void    ApplyX(TF1 *f);
   static Int_t    CalculateScanfFields(const char *fmt);
   void    ComputeRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax) const override;
   Double_t        GetErrorX(Int_t bin)     const override;
   Double_t        GetErrorY(Int_t bin)     const override;
   Double_t        GetErrorXhigh(Int_t bin) const override;
   Double_t        GetErrorXlow(Int_t bin)  const override;
   Double_t        GetErrorYhigh(Int_t bin) const override;
   Double_t        GetErrorYlow(Int_t bin)  const override;
   Double_t       *GetEX() const override {return fEX;}
   Double_t       *GetEY() const override {return fEY;}
   Int_t   Merge(TCollection* list) override;
   void    Print(Option_t *chopt="") const override;
   void    SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void    Scale(Double_t c1=1., Option_t *option="y") override; // *MENU*
   virtual void    SetPointError(Double_t ex, Double_t ey);  // *MENU
   virtual void    SetPointError(Int_t i, Double_t ex, Double_t ey);

   ClassDefOverride(TGraphErrors,3)  //A graph with error bars
};

inline Double_t **TGraphErrors::Allocate(Int_t size) {
   return AllocateArrays(4, size);
}

#endif
