// @(#)root/hist:$Name:  $:$Id: TF1.h,v 1.57 2006/07/03 16:10:45 brun Exp $
// Author: Rene Brun   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- F1.h

#ifndef ROOT_TF1
#define ROOT_TF1



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TF1                                                                  //
//                                                                      //
// The Parametric 1-D function                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFormula
#include "TFormula.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif
#ifndef ROOT_TMethodCall
#include "TMethodCall.h"
#endif

class TF1;
class TH1;
class TAxis;

class TF1 : public TFormula, public TAttLine, public TAttFill, public TAttMarker {

protected:
   Double_t    fXmin;        //Lower bounds for the range
   Double_t    fXmax;        //Upper bounds for the range
   Int_t       fNpx;         //Number of points used for the graphical representation
   Int_t       fType;        //(=0 for standard functions, 1 if pointer to function)
   Int_t       fNpfits;      //Number of points used in the fit
   Int_t       fNDF;         //Number of degrees of freedom in the fit
   Int_t       fNsave;       //Number of points used to fill array fSave
   Double_t    fChisquare;   //Function fit chisquare
   Double_t    *fIntegral;   //![fNpx] Integral of function binned on fNpx bins
   Double_t    *fParErrors;  //[fNpar] Array of errors of the fNpar parameters
   Double_t    *fParMin;     //[fNpar] Array of lower limits of the fNpar parameters
   Double_t    *fParMax;     //[fNpar] Array of upper limits of the fNpar parameters
   Double_t    *fSave;       //[fNsave] Array of fNsave function values
   Double_t    *fAlpha;      //!Array alpha. for each bin in x the deconvolution r of fIntegral
   Double_t    *fBeta;       //!Array beta.  is approximated by x = alpha +beta*r *gamma*r**2
   Double_t    *fGamma;      //!Array gamma.
   TObject     *fParent;     //!Parent object hooking this function (if one)
   TH1         *fHistogram;  //!Pointer to histogram used for visualisation
   Double_t     fMaximum;    //Maximum value for plotting
   Double_t     fMinimum;    //Minimum value for plotting
   TMethodCall *fMethodCall; //!Pointer to MethodCall in case of interpreted function
   Double_t (*fFunction) (Double_t *, Double_t *);   //!Pointer to function

   static Bool_t fgAbsValue;  //use absolute value of function when computing integral
   static Bool_t fgRejectPoint;  //True if point must be rejected in a fit
   static TF1   *fgCurrent;   //pointer to current function being processed
         
public:
    // TF1 status bits
    enum {
       kNotDraw     = BIT(9)  // don't draw the function when in a TH1
    };
   TF1();
   TF1(const char *name, const char *formula, Double_t xmin=0, Double_t xmax=1);
   TF1(const char *name, Double_t xmin, Double_t xmax, Int_t npar);
   TF1(const char *name, void *fcn, Double_t xmin, Double_t xmax, Int_t npar);
#ifndef __CINT__
   TF1(const char *name, Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin=0, Double_t xmax=1, Int_t npar=0);
   TF1(const char *name, Double_t (*fcn)(const Double_t *, const Double_t *), Double_t xmin=0, Double_t xmax=1, Int_t npar=0);
#endif
   TF1(const TF1 &f1);
   TF1& operator=(const TF1 &rhs);
   virtual   ~TF1();
   virtual void     Browse(TBrowser *b);
   virtual void     Copy(TObject &f1) const;
   virtual Double_t Derivative (Double_t x, Double_t *params=0, Double_t epsilon=0.001) const;
   virtual Double_t Derivative2(Double_t x, Double_t *params=0, Double_t epsilon=0.001) const;
   virtual Double_t Derivative3(Double_t x, Double_t *params=0, Double_t epsilon=0.001) const;
   static  Double_t DerivativeError();
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual TF1     *DrawCopy(Option_t *option="") const;
   virtual void     DrawDerivative(Option_t *option="al"); // *MENU*
   virtual void     DrawIntegral(Option_t *option="al");   // *MENU*
   virtual void     DrawF1(const char *formula, Double_t xmin, Double_t xmax, Option_t *option="");
   virtual Double_t Eval(Double_t x, Double_t y=0, Double_t z=0, Double_t t=0) const;
   virtual Double_t EvalPar(const Double_t *x, const Double_t *params=0);
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void     FixParameter(Int_t ipar, Double_t value);
       Double_t     GetChisquare() const {return fChisquare;}
           TH1     *GetHistogram() const;
   virtual Double_t GetMaximum(Double_t xmin=0, Double_t xmax=0) const;
   virtual Double_t GetMinimum(Double_t xmin=0, Double_t xmax=0) const;
   virtual Double_t GetMaximumX(Double_t xmin=0, Double_t xmax=0) const;
   virtual Double_t GetMinimumX(Double_t xmin=0, Double_t xmax=0) const;
   virtual Int_t    GetNDF() const;
   virtual Int_t    GetNpx() const {return fNpx;}
    TMethodCall    *GetMethodCall() const {return fMethodCall;}
   virtual Int_t    GetNumberFreeParameters() const;
   virtual Int_t    GetNumberFitPoints() const {return fNpfits;}
   virtual char    *GetObjectInfo(Int_t px, Int_t py) const;
        TObject    *GetParent() const {return fParent;}
   virtual Double_t GetParError(Int_t ipar) const;
   virtual Double_t *GetParErrors() const {return fParErrors;}
   virtual void     GetParLimits(Int_t ipar, Double_t &parmin, Double_t &parmax) const;
   virtual Double_t GetProb() const;
   virtual Int_t    GetQuantiles(Int_t nprobSum, Double_t *q, const Double_t *probSum); 
   virtual Double_t GetRandom();
   virtual Double_t GetRandom(Double_t xmin, Double_t xmax);
   virtual void     GetRange(Double_t &xmin, Double_t &xmax) const;
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax) const;
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const;
   virtual Double_t GetSave(const Double_t *x);
   virtual Double_t GetX(Double_t y, Double_t xmin=0, Double_t xmax=0) const;
   virtual Double_t GetXmin() const {return fXmin;}
   virtual Double_t GetXmax() const {return fXmax;}
   TAxis           *GetXaxis() const ;
   TAxis           *GetYaxis() const ;
   TAxis           *GetZaxis() const ;
   virtual void     GradientPar(const Double_t *x, Double_t *grad, Double_t eps=0.01);
   virtual void     InitArgs(const Double_t *x, const Double_t *params);
   static  void     InitStandardFunctions();
   virtual Double_t Integral(Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=1e-12);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsilon=1e-12);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsilon=1e-12);
   //virtual Double_t IntegralFast(const TGraph *g, Double_t a, Double_t b, Double_t *params=0);
   virtual Double_t IntegralFast(Int_t num, Double_t *x, Double_t *w, Double_t a, Double_t b, Double_t *params=0);
   virtual Double_t IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Int_t minpts, Int_t maxpts, Double_t epsilon, Double_t &relerr,Int_t &nfnevl, Int_t &ifail);
   virtual Double_t IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Double_t epsilon, Double_t &relerr);
   virtual Bool_t   IsInside(const Double_t *x) const;
   Double_t         MinimBrent(Int_t type, Double_t &xmin, Double_t &xmax, Double_t xmiddle, Double_t fy, Bool_t &ok) const;
   Double_t         MinimStep(Int_t type, Double_t &xmin, Double_t &xmax, Double_t fy) const;
   virtual void     Paint(Option_t *option="");
   virtual void     Print(Option_t *option="") const;
   virtual void     ReleaseParameter(Int_t ipar);
   virtual void     Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax);
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetChisquare(Double_t chi2) {fChisquare = chi2;}
   virtual void     SetFunction(Double_t (*fcn)(Double_t *, Double_t *)) { fFunction = fcn;}
   virtual void     SetMaximum(Double_t maximum=-1111); // *MENU*
   virtual void     SetMinimum(Double_t minimum=-1111); // *MENU*
   virtual void     SetNDF(Int_t ndf);
   virtual void     SetNumberFitPoints(Int_t npfits) {fNpfits = npfits;}
   virtual void     SetNpx(Int_t npx=100); // *MENU*
   virtual void     SetParError(Int_t ipar, Double_t error);
   virtual void     SetParErrors(const Double_t *errors);
   virtual void     SetParLimits(Int_t ipar, Double_t parmin, Double_t parmax);
   virtual void     SetParent(TObject *p=0) {fParent = p;}
   virtual void     SetRange(Double_t xmin, Double_t xmax); // *MENU*
   virtual void     SetRange(Double_t xmin, Double_t ymin,  Double_t xmax, Double_t ymax);
   virtual void     SetRange(Double_t xmin, Double_t ymin, Double_t zmin,  Double_t xmax, Double_t ymax, Double_t zmax);
   virtual void     SetSavedPoint(Int_t point, Double_t value);
   virtual void     SetTitle(const char *title=""); // *MENU*
   virtual void     Update();

   static  TF1     *GetCurrent();
   static  void     AbsValue(Bool_t reject=kTRUE);
   static  void     RejectPoint(Bool_t reject=kTRUE);
   static  Bool_t   RejectedPoint();
   static  void     SetCurrent(TF1 *f1);

   //Moments
   virtual Double_t Moment(Double_t n, Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=0.000001);
   virtual Double_t CentralMoment(Double_t n, Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=0.000001);
   virtual Double_t Mean(Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=0.000001) {return Moment(1,a,b,params,epsilon);}
   virtual Double_t Variance(Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=0.000001) {return CentralMoment(2,a,b,params,epsilon);}

   //some useful static utility functions to compute sampling points for Integral
   //static  void     CalcGaussLegendreSamplingPoints(TGraph *g, Double_t eps=3.0e-11);
   //static  TGraph  *CalcGaussLegendreSamplingPoints(Int_t num=21, Double_t eps=3.0e-11);
   static  void     CalcGaussLegendreSamplingPoints(Int_t num, Double_t *x, Double_t *w, Double_t eps=3.0e-11);
        
   ClassDef(TF1,7)  //The Parametric 1-D function
};

inline void TF1::SetRange(Double_t xmin, Double_t,  Double_t xmax, Double_t)
   { TF1::SetRange(xmin, xmax); }
inline void TF1::SetRange(Double_t xmin, Double_t, Double_t,  Double_t xmax, Double_t, Double_t)
   { TF1::SetRange(xmin, xmax); }

#endif
