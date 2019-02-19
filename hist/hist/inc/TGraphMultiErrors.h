// @(#)root/hist:$Id$
// Author: Simon Spies 18/02/19
 
/*************************************************************************
 * Copyright (C) 2018-2019, Simon Spies.                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGraphMultiErrors
#define ROOT_TGraphMultiErrors

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGraphMultiErrors                                                    //
//                                                                      //
// a Graph with asymmetric error bars and multiple y error dimensions   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGraph.h"

class TGraphMultiErrors : public TGraph {

protected:
    Int_t      fNErrorDimensions;
    Double_t*  fExL;     //<[fNpoints] array of X low errors
    Double_t*  fExH;     //<[fNpoints] array of X high errors
    Double_t** fEyL;     //<[fNErrorDimensions][fNpoints] two dimensional array of Y low errors
    Double_t** fEyH;     //<[fNErrorDimensions][fNpoints] two dimensional array of Y high errors
    Double_t*  fEyLSum;  //<[fNpoints] array of summed Y low errors for fitting
    Double_t*  fEyHSum;  //<[fNpoints] array of summed Y high errors for fitting
    Int_t      fSumErrorsMode; // How y errors are summed: kOnlyFirst = Only Fist; kSquareSum = Squared Sum; kSum = Absolute Addition
    TAttFill*  fAttFill; //<[fNErrorDimensions] the AttFill attributes of the different error dimensions
    TAttLine*  fAttLine; //<[fNErrorDimensions] the AttLine attributes of the different error dimensions


    virtual void       SwapPoints(Int_t pos1, Int_t pos2);
    virtual Double_t** Allocate(Int_t size);
    virtual void       CopyAndRelease(Double_t** newarrays, Int_t ibegin, Int_t iend, Int_t obegin);
    virtual void       CopyAndReleaseY(Double_t** newarrays, Int_t ibegin, Int_t iend, Int_t obegin);
    virtual Bool_t     CopyPoints(Double_t** arrays, Int_t ibegin, Int_t iend, Int_t obegin);
    virtual Bool_t     CopyPointsY(Double_t** arrays, Int_t ibegin, Int_t iend, Int_t obegin);
    Bool_t             CtorAllocate();
    virtual void       FillZero(Int_t begin, Int_t end, Bool_t from_ctor = kTRUE);
    virtual void       FillZeroY(Int_t begin, Int_t end);
    virtual Bool_t     DoMerge(const TGraph* tg);
    virtual void       CalcYErrorSum();

public:
    enum ESummationModes {
	kOnlyFirst = 0, ///< Only take errors from first dimension
	kSquareSum = 1, ///< Calculate the square sum of all errors
	kSum       = 2  ///< Calculate the sum of all errors
    };

    TGraphMultiErrors();
    TGraphMultiErrors(Int_t n, Int_t dim = 1);

    TGraphMultiErrors(Int_t n, const Float_t*  x, const Float_t*  y, const Float_t*  exL = NULL, const Float_t*  exH = NULL, const Float_t*  eyL = NULL, const Float_t*  eyH = NULL, Int_t m = kOnlyFirst);
    TGraphMultiErrors(Int_t n, const Double_t* x, const Double_t* y, const Double_t* exL = NULL, const Double_t* exH = NULL, const Double_t* eyL = NULL, const Double_t* eyH = NULL, Int_t m = kOnlyFirst);
    TGraphMultiErrors(Int_t n, Int_t dim, const Float_t*  x, const Float_t*  y, const Float_t*  exL = NULL, const Float_t*  exH = NULL, Float_t**  eyL = NULL, Float_t**  eyH = NULL, Int_t m = kOnlyFirst);
    TGraphMultiErrors(Int_t n, Int_t dim, const Double_t* x, const Double_t* y, const Double_t* exL = NULL, const Double_t* exH = NULL, Double_t** eyL = NULL, Double_t** eyH = NULL, Int_t m = kOnlyFirst);

    TGraphMultiErrors(const TVectorF& tvX, const TVectorF& tvY, const TVectorF& tvExL, const TVectorF& tvExH, const TVectorF& tvEyL, const TVectorF& tvEyH, Int_t m = kOnlyFirst);
    TGraphMultiErrors(const TVectorD& tvX, const TVectorD& tvY, const TVectorD& tvExL, const TVectorD& tvExH, const TVectorD& tvEyL, const TVectorD& tvEyH, Int_t m = kOnlyFirst);
    TGraphMultiErrors(Int_t dim, const TVectorF& tvX, const TVectorF& tvY, const TVectorF& tvExL, const TVectorF& tvExH, const TVectorF* tvEyL, const TVectorF* tvEyH, Int_t m = kOnlyFirst);
    TGraphMultiErrors(Int_t dim, const TVectorD& tvX, const TVectorD& tvY, const TVectorD& tvExL, const TVectorD& tvExH, const TVectorD* tvEyL, const TVectorD* tvEyH, Int_t m = kOnlyFirst);

    TGraphMultiErrors(const TGraphMultiErrors& tgde);
    TGraphMultiErrors& operator = (const TGraphMultiErrors& tgde);

    TGraphMultiErrors(const TH1* th, Int_t dim = 1);
    TGraphMultiErrors(const TH1* pass, const TH1* total, Int_t dim = 1, Option_t* option = "");

    virtual ~TGraphMultiErrors();


    virtual void  Apply(TF1* f);
    virtual void  BayesDivide(const TH1* pass, const TH1* total, Option_t* opt = "");
    virtual void  Divide(const TH1* pass, const TH1* total, Option_t* opt = "cp");
    virtual void  ComputeRange(Double_t& xmin, Double_t& ymin, Double_t& xmax, Double_t& ymax) const;

    Double_t      GetPointX(Int_t i)                const;
    Double_t      GetPointY(Int_t i)                const;

    Double_t      GetErrorX(Int_t i)                const;
    Double_t      GetErrorY(Int_t i)                const;
    Double_t      GetErrorY(Int_t i, Int_t dim)     const;

    Double_t      GetErrorXlow(Int_t i)             const;
    Double_t      GetErrorXhigh(Int_t i)            const;
    Double_t      GetErrorYlow(Int_t i)             const;
    Double_t      GetErrorYhigh(Int_t i)            const;
    Double_t      GetErrorYlow(Int_t i, Int_t dim)  const;
    Double_t      GetErrorYhigh(Int_t i, Int_t dim) const;

    Double_t*     GetEXlow()                        const { return fExL; }
    Double_t*     GetEXhigh()                       const { return fExH; }
    Double_t*     GetEYlow()                        const { return fEyLSum; }
    Double_t*     GetEYhigh()                       const { return fEyHSum; }
    Double_t*     GetEYlow(Int_t dim);
    Double_t*     GetEYhigh(Int_t dim);

    TAttFill*     GetAttFill(Int_t dim)             const;
    TAttLine*     GetAttLine(Int_t dim)             const;

    using TAttFill::GetFillColor;
    using TAttFill::GetFillStyle;

    Color_t       GetFillColor(Int_t dim)           const;
    Style_t       GetFillStyle(Int_t dim)           const;

    using TAttLine::GetLineColor;
    using TAttLine::GetLineStyle;
    using TAttLine::GetLineWidth;

    Color_t       GetLineColor(Int_t dim)           const;
    Style_t       GetLineStyle(Int_t dim)           const;
    Width_t       GetLineWidth(Int_t dim)           const;


    Int_t         GetSumErrorsMode()                const { return fSumErrorsMode; }
    Int_t         GetNErrorDimensions()             const { return fNErrorDimensions; }

    virtual char* GetObjectInfo(Int_t px, Int_t py) const;
    virtual void  Print(Option_t* chopt = "") const;
    virtual void  SavePrimitive(std::ostream& out, Option_t* option = "");

    virtual void  SetPointX(Int_t i, Double_t x);
    virtual void  SetPointY(Int_t i, Double_t y);

    virtual void  SetPointError(Double_t exL, Double_t exH, Double_t eyL1, Double_t eyH1, Double_t eyL2 = 0., Double_t eyH2 = 0., Double_t eyL3 = 0., Double_t eyH3 = 0.); // *MENU*
    virtual void  SetPointError(Int_t i, Double_t exL, Double_t exH, Double_t* eyL, Double_t* eyH);

    virtual void  SetPointEX(Int_t i, Double_t exL, Double_t exH);
    virtual void  SetPointEXL(Int_t i, Double_t exL);
    virtual void  SetPointEXH(Int_t i, Double_t exH);
    virtual void  SetPointEY(Int_t i, Double_t* eyL, Double_t* eyH);
    virtual void  SetPointEYL(Int_t i, Double_t* eyL);
    virtual void  SetPointEYH(Int_t i, Double_t* eyH);
    virtual void  SetPointEY(Int_t i, Int_t dim, Double_t eyL, Double_t eyH);
    virtual void  SetPointEYL(Int_t i, Int_t dim, Double_t eyL);
    virtual void  SetPointEYH(Int_t i, Int_t dim, Double_t eyH);

    virtual void  SetDimensionEY(Int_t dim, Double_t* eyL, Double_t* eyH);
    virtual void  SetDimensionEYL(Int_t dim, Double_t* eyL);
    virtual void  SetDimensionEYH(Int_t dim, Double_t* eyH);

    virtual void  SetSumErrorsMode(Int_t m) { fSumErrorsMode = m; }
    virtual void  SetNErrorDimensions(Int_t dim);

    virtual void  SetAttFill(Int_t dim, TAttFill* taf);
    virtual void  SetAttLine(Int_t dim, TAttLine* tal);

    using TAttFill::SetFillColor;
    using TAttFill::SetFillColorAlpha;
    using TAttFill::SetFillStyle;

    virtual void  SetFillColor(Int_t dim, Color_t fcolor);
    virtual void  SetFillColorAlpha(Int_t dim, Color_t fcolor, Float_t falpha);
    virtual void  SetFillStyle(Int_t dim, Style_t fstyle);

    using TAttLine::SetLineColor;
    using TAttLine::SetLineColorAlpha;
    using TAttLine::SetLineStyle;
    using TAttLine::SetLineWidth;

    virtual void  SetLineColor(Int_t dim, Color_t lcolor);
    virtual void  SetLineColorAlpha(Int_t dim, Color_t lcolor, Float_t lalpha);
    virtual void  SetLineStyle(Int_t dim, Style_t lstyle);
    virtual void  SetLineWidth(Int_t dim, Width_t lwidth);

    ClassDef(TGraphMultiErrors, 1)  //A Graph with asymmetric error bars and multiple y error dimensions
};

inline Double_t** TGraphMultiErrors::Allocate(Int_t size) {
    return AllocateArrays(2*fNErrorDimensions + 4, size);
}
#endif // ROOT_TGraphMultiErrors
