// @(#)root/hist:$Name:  $:$Id: TF1.h,v 1.14 2001/07/09 17:20:55 brun Exp $
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

public:
    // TF1 status bits
    enum {
       kNotDraw     = BIT(9)  // don't draw the function when in a TH1
    };
   TF1();
   TF1(const char *name, const char *formula, Double_t xmin=0, Double_t xmax=1);
   TF1(const char *name, Double_t xmin, Double_t xmax, Int_t npar);
   TF1(const char *name, void *fcn, Double_t xmin, Double_t xmax, Int_t npar);
   TF1(const char *name, Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin=0, Double_t xmax=1, Int_t npar=0);
   TF1(const TF1 &f1);
   virtual   ~TF1();
   virtual void     Browse(TBrowser *b);
   virtual void     Copy(TObject &f1);
   virtual Double_t Derivative(Double_t x, Double_t *params=0, Double_t epsilon=0);
   virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   virtual void     Draw(Option_t *option="");
   virtual TF1     *DrawCopy(Option_t *option="");
   virtual void     DrawF1(const char *formula, Double_t xmin, Double_t xmax, Option_t *option="");
   virtual void     DrawPanel(); // *MENU*
   virtual Double_t Eval(Double_t x, Double_t y=0, Double_t z=0);
   virtual Double_t EvalPar(const Double_t *x, const Double_t *params=0);
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void     FixParameter(Int_t ipar, Double_t value);
       Double_t     GetChisquare() const {return fChisquare;}
           TH1     *GetHistogram() const;
          Int_t     GetNDF() const;
          Int_t     GetNpx() const {return fNpx;}
    TMethodCall    *GetMethodCall() const {return fMethodCall;}
          Int_t     GetNumberFitPoints() const {return fNpfits;}
   virtual char    *GetObjectInfo(Int_t px, Int_t py) const;
        TObject    *GetParent() const {return fParent;}
       Double_t     GetParError(Int_t ipar) const {return fParErrors[ipar];}
       Double_t    *GetParErrors() const {return fParErrors;}
   virtual void     GetParLimits(Int_t ipar, Double_t &parmin, Double_t &parmax);
   virtual Double_t GetProb() const {return TMath::Prob(fChisquare,fNpfits-fNpar);}
   virtual Double_t GetRandom();
   virtual void     GetRange(Double_t &xmin, Double_t &xmax);
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &xmax, Double_t &ymax);
   virtual void     GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax);
   virtual Double_t GetSave(const Double_t *x);
        Double_t    GetXmin() const {return fXmin;}
        Double_t    GetXmax() const {return fXmax;}
   virtual void     InitArgs(const Double_t *x, const Double_t *params);
   static  void     InitStandardFunctions();
   virtual Double_t Integral(Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=0.000001);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsilon=0.000001);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsilon=0.000001);
   virtual Double_t IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Double_t epsilon, Double_t &relerr);
   virtual void     Paint(Option_t *option="");
   virtual void     Print(Option_t *option="") const;
   virtual void     ReleaseParameter(Int_t ipar);
   virtual void     Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax);
   virtual void     SavePrimitive(ofstream &out, Option_t *option);
   virtual void     SetChisquare(Double_t chi2) {fChisquare = chi2;}
   virtual void     SetMaximum(Double_t maximum=-1111) {fMaximum=maximum;} // *MENU*
   virtual void     SetMinimum(Double_t minimum=-1111) {fMinimum=minimum;} // *MENU*
   virtual void     SetNDF(Int_t ndf);
   virtual void     SetNumberFitPoints(Int_t npfits) {fNpfits = npfits;}
   virtual void     SetNpx(Int_t npx=100); // *MENU*
   virtual void     SetParError(Int_t ipar, Double_t error) {fParErrors[ipar] = error;}
   virtual void     SetParLimits(Int_t ipar, Double_t parmin, Double_t parmax);
   virtual void     SetParent(TObject *p=0) {fParent = p;}
   virtual void     SetRange(Double_t xmin, Double_t xmax); // *MENU*
   virtual void     SetRange(Double_t xmin, Double_t ymin,  Double_t xmax, Double_t ymax);
   virtual void     SetRange(Double_t xmin, Double_t ymin, Double_t zmin,  Double_t xmax, Double_t ymax, Double_t zmax);
   virtual void     Update();

   ClassDef(TF1,7)  //The Parametric 1-D function
};

inline void TF1::SetRange(Double_t xmin, Double_t,  Double_t xmax, Double_t)
   { TF1::SetRange(xmin, xmax); }
inline void TF1::SetRange(Double_t xmin, Double_t, Double_t,  Double_t xmax, Double_t, Double_t)
   { TF1::SetRange(xmin, xmax); }

#endif
