// @(#)root/hist:$Id$
// Author: Dave Morrison  30/06/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphBentErrors
#define ROOT_TGraphBentErrors

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphBentErrors                                                     //
//                                                                      //
// a Graph with bent, asymmetric error bars                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph.h"

class TGraphBentErrors : public TGraph {

protected:
   Double_t    *fEXlow;        ///<[fNpoints] array of X low errors
   Double_t    *fEXhigh;       ///<[fNpoints] array of X high errors
   Double_t    *fEYlow;        ///<[fNpoints] array of Y low errors
   Double_t    *fEYhigh;       ///<[fNpoints] array of Y high errors

   Double_t    *fEXlowd;       ///<[fNpoints] array of X low displacements
   Double_t    *fEXhighd;      ///<[fNpoints] array of X high displacements
   Double_t    *fEYlowd;       ///<[fNpoints] array of Y low displacements
   Double_t    *fEYhighd;      ///<[fNpoints] array of Y high displacements

   void       SwapPoints(Int_t pos1, Int_t pos2) override;

   Double_t** Allocate(Int_t size) override;
   void       CopyAndRelease(Double_t **newarrays,
                                     Int_t ibegin, Int_t iend, Int_t obegin) override;
   Bool_t     CopyPoints(Double_t **arrays, Int_t ibegin, Int_t iend,
                                 Int_t obegin) override;
   Bool_t             CtorAllocate();
   void       FillZero(Int_t begin, Int_t end,
                               Bool_t from_ctor = kTRUE) override;
   Bool_t     DoMerge(const TGraph * g) override;


public:
   TGraphBentErrors();
   TGraphBentErrors(Int_t n);
   TGraphBentErrors(Int_t n,
                    const Float_t *x, const Float_t *y,
                    const Float_t *exl=0, const Float_t *exh=0,
                    const Float_t *eyl=0, const Float_t *eyh=0,
                    const Float_t *exld=0, const Float_t *exhd=0,
                    const Float_t *eyld=0, const Float_t *eyhd=0);
   TGraphBentErrors(Int_t n,
                    const Double_t *x, const Double_t *y,
                    const Double_t *exl=0, const Double_t *exh=0,
                    const Double_t *eyl=0, const Double_t *eyh=0,
                    const Double_t *exld=0, const Double_t *exhd=0,
                    const Double_t *eyld=0, const Double_t *eyhd=0);
   TGraphBentErrors(const TGraphBentErrors &gr);
   ~TGraphBentErrors() override;
   void    Apply(TF1 *f) override;
   void    ComputeRange(Double_t &xmin, Double_t &ymin,
                                Double_t &xmax, Double_t &ymax) const override;
   Double_t        GetErrorX(Int_t bin)     const override;
   Double_t        GetErrorY(Int_t bin)     const override;
   Double_t        GetErrorXlow(Int_t bin)  const override;
   Double_t        GetErrorXhigh(Int_t bin) const override;
   Double_t        GetErrorYlow(Int_t bin)  const override;
   Double_t        GetErrorYhigh(Int_t bin) const override;
   Double_t       *GetEXlow()   const override {return fEXlow;}
   Double_t       *GetEXhigh()  const override {return fEXhigh;}
   Double_t       *GetEYlow()   const override {return fEYlow;}
   Double_t       *GetEYhigh()  const override {return fEYhigh;}
   Double_t       *GetEXlowd()  const override {return fEXlowd;}
   Double_t       *GetEXhighd() const override {return fEXhighd;}
   Double_t       *GetEYlowd()  const override {return fEYlowd;}
   Double_t       *GetEYhighd() const override {return fEYhighd;}
   void    Print(Option_t *chopt="") const override;
   void    SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void    Scale(Double_t c1=1., Option_t *option="y") override; // *MENU*
   virtual void    SetPointError(Double_t exl, Double_t exh,
                                 Double_t eyl, Double_t eyh,
                                 Double_t exld=0, Double_t exhd=0,
                                 Double_t eyld=0, Double_t eyhd=0); // *MENU*
   virtual void    SetPointError(Int_t i,
                                 Double_t exl, Double_t exh,
                                 Double_t eyl, Double_t eyh,
                                 Double_t exld=0, Double_t exhd=0,
                                 Double_t eyld=0, Double_t eyhd=0);

   ClassDefOverride(TGraphBentErrors,1)  //A graph with bent, asymmetric error bars
};

inline Double_t **TGraphBentErrors::Allocate(Int_t size) {
   return AllocateArrays(10, size);
}

#endif
