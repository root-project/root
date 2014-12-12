// @(#)root/hist:$Id$
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

#ifndef ROOT_Math_ParamFunctor
#include "Math/ParamFunctor.h"
#endif

class TF1;
class TH1;
class TAxis;
class TMethodCall;

namespace ROOT { 
   namespace Fit { 
      class FitResult; 
   }
}

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
   void        *fCintFunc;              //! pointer to interpreted function class
   ROOT::Math::ParamFunctor fFunctor;   //! Functor object to wrap any C++ callable object

   static Bool_t fgAbsValue;  //use absolute value of function when computing integral
   static Bool_t fgRejectPoint;  //True if point must be rejected in a fit
   static TF1   *fgCurrent;   //pointer to current function being processed

   void CreateFromFunctor(const char *name, Int_t npar);
   void CreateFromCintClass(const char *name, void * ptr, Double_t xmin, Double_t xmax, Int_t npar, const char * cname, const char * fname);

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

   // Constructors using functors (compiled mode only)
   TF1(const char *name, ROOT::Math::ParamFunctor f, Double_t xmin = 0, Double_t xmax = 1, Int_t npar = 0);

   // Template constructors from any  C++ callable object,  defining  the operator() (double * , double *)
   // and returning a double.
   // The class name is not needed when using compile code, while it is required when using
   // interpreted code via the specialized constructor with void *.
   // An instance of the C++ function class or its pointer can both be used. The former is reccomended when using
   // C++ compiled code, but if CINT compatibility is needed, then a pointer to the function class must be used.
   // xmin and xmax specify the plotting range,  npar is the number of parameters.
   // See the tutorial math/exampleFunctor.C for an example of using this constructor
   template <typename Func>
   TF1(const char *name, Func f, Double_t xmin, Double_t xmax, Int_t npar, const char * = 0  ) :
      TFormula(),
      TAttLine(),
      TAttFill(),
      TAttMarker(),
      fXmin      ( xmin ),
      fXmax      ( xmax ),
      fNpx       ( 100 ),
      fType      ( 1 ),
      fNpfits    ( 0 ),
      fNDF       ( 0 ),
      fNsave     ( 0 ),
      fChisquare ( 0 ),
      fIntegral  ( 0 ),
      fParErrors ( 0 ),
      fParMin    ( 0 ),
      fParMax    ( 0 ),
      fSave      ( 0 ),
      fAlpha     ( 0 ),
      fBeta      ( 0 ),
      fGamma     ( 0 ),
      fParent    ( 0 ),
      fHistogram ( 0 ),
      fMaximum   ( -1111 ),
      fMinimum   ( -1111 ),
      fMethodCall ( 0),
      fCintFunc  ( 0 ),
      fFunctor( ROOT::Math::ParamFunctor(f) )
   {
      CreateFromFunctor(name, npar);
   }

   // Template constructors from a pointer to any C++ class of type PtrObj with a specific member function of type
   // MemFn.
   // The member function must have the signature of  (double * , double *) and returning a double.
   // The class name and the method name are not needed when using compile code
   // (the member function pointer is used in this case), while they are required when using interpreted
   // code via the specialized constructor with void *.
   // xmin and xmax specify the plotting range,  npar is the number of parameters.
   // See the tutorial math/exampleFunctor.C for an example of using this constructor
   template <class PtrObj, typename MemFn>
   TF1(const char *name, const  PtrObj& p, MemFn memFn, Double_t xmin, Double_t xmax, Int_t npar, const char * = 0, const char * = 0) :
      TFormula(),
      TAttLine(),
      TAttFill(),
      TAttMarker(),
      fXmin      ( xmin ),
      fXmax      ( xmax ),
      fNpx       ( 100 ),
      fType      ( 1 ),
      fNpfits    ( 0 ),
      fNDF       ( 0 ),
      fNsave     ( 0 ),
      fChisquare ( 0 ),
      fIntegral  ( 0 ),
      fParErrors ( 0 ),
      fParMin    ( 0 ),
      fParMax    ( 0 ),
      fSave      ( 0 ),
      fAlpha     ( 0 ),
      fBeta      ( 0 ),
      fGamma     ( 0 ),
      fParent    ( 0 ),
      fHistogram ( 0 ),
      fMaximum   ( -1111 ),
      fMinimum   ( -1111 ),
      fMethodCall( 0 ),
      fCintFunc  ( 0 ),
      fFunctor   ( ROOT::Math::ParamFunctor(p,memFn) )
   {
      CreateFromFunctor(name, npar);
   }

   // constructor used by CINT
   TF1(const char *name, void *ptr, Double_t xmin, Double_t xmax, Int_t npar, const char *className );
   TF1(const char *name, void *ptr, void *,Double_t xmin, Double_t xmax, Int_t npar, const char *className, const char *methodName = 0);

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
   virtual TObject *DrawDerivative(Option_t *option="al"); // *MENU*
   virtual TObject *DrawIntegral(Option_t *option="al");   // *MENU*
   virtual void     DrawF1(const char *formula, Double_t xmin, Double_t xmax, Option_t *option="");
   virtual Double_t Eval(Double_t x, Double_t y=0, Double_t z=0, Double_t t=0) const;
   virtual Double_t EvalPar(const Double_t *x, const Double_t *params=0);
   // for using TF1 as a callable object (functor)
   virtual Double_t operator()(Double_t x, Double_t y=0, Double_t z = 0, Double_t t = 0) const; 
   virtual Double_t operator()(const Double_t *x, const Double_t *params=0);  
   virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual void     FixParameter(Int_t ipar, Double_t value);
       Double_t     GetChisquare() const {return fChisquare;}
           TH1     *GetHistogram() const;
   virtual Double_t GetMaximum(Double_t xmin=0, Double_t xmax=0, Double_t epsilon = 1.E-10, Int_t maxiter = 100, Bool_t logx = false) const;
   virtual Double_t GetMinimum(Double_t xmin=0, Double_t xmax=0, Double_t epsilon = 1.E-10, Int_t maxiter = 100, Bool_t logx = false) const;
   virtual Double_t GetMaximumX(Double_t xmin=0, Double_t xmax=0, Double_t epsilon = 1.E-10, Int_t maxiter = 100, Bool_t logx = false) const;
   virtual Double_t GetMinimumX(Double_t xmin=0, Double_t xmax=0, Double_t epsilon = 1.E-10, Int_t maxiter = 100, Bool_t logx = false) const;
   virtual Double_t GetMaximumStored() const {return fMaximum;}
   virtual Double_t GetMinimumStored() const {return fMinimum;}
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
   virtual Double_t GetX(Double_t y, Double_t xmin=0, Double_t xmax=0, Double_t epsilon = 1.E-10, Int_t maxiter = 100, Bool_t logx = false) const;
   virtual Double_t GetXmin() const {return fXmin;}
   virtual Double_t GetXmax() const {return fXmax;}
   TAxis           *GetXaxis() const ;
   TAxis           *GetYaxis() const ;
   TAxis           *GetZaxis() const ;
   virtual Double_t GradientPar(Int_t ipar, const Double_t *x, Double_t eps=0.01);
   virtual void     GradientPar(const Double_t *x, Double_t *grad, Double_t eps=0.01);
   virtual void     InitArgs(const Double_t *x, const Double_t *params);
   static  void     InitStandardFunctions();
   virtual Double_t Integral(Double_t a, Double_t b, const Double_t *params=0, Double_t epsilon=1e-12);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t epsilon=1e-12);
   virtual Double_t Integral(Double_t ax, Double_t bx, Double_t ay, Double_t by, Double_t az, Double_t bz, Double_t epsilon=1e-12);
   virtual Double_t IntegralError(Double_t a, Double_t b, const Double_t *params=0, const Double_t *covmat=0, Double_t epsilon=1e-12);
   virtual Double_t IntegralError(Int_t n, const Double_t * a, const Double_t * b, const Double_t *params=0, const Double_t *covmat=0, Double_t epsilon=1e-12);
   //virtual Double_t IntegralFast(const TGraph *g, Double_t a, Double_t b, Double_t *params=0);
   virtual Double_t IntegralFast(Int_t num, Double_t *x, Double_t *w, Double_t a, Double_t b, Double_t *params=0, Double_t epsilon=1e-12);
   virtual Double_t IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Int_t minpts, Int_t maxpts, Double_t epsilon, Double_t &relerr,Int_t &nfnevl, Int_t &ifail);
   virtual Double_t IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Double_t epsilon, Double_t &relerr);
   virtual Bool_t   IsInside(const Double_t *x) const;
   virtual void     Paint(Option_t *option="");
   virtual void     Print(Option_t *option="") const;
   virtual void     ReleaseParameter(Int_t ipar);
   virtual void     Save(Double_t xmin, Double_t xmax, Double_t ymin, Double_t ymax, Double_t zmin, Double_t zmax);
   virtual void     SavePrimitive(ostream &out, Option_t *option = "");
   virtual void     SetChisquare(Double_t chi2) {fChisquare = chi2;}
   virtual void     SetFitResult(const ROOT::Fit::FitResult & result, const Int_t * indpar = 0);
   template <class PtrObj, typename MemFn> 
   void SetFunction( PtrObj& p, MemFn memFn );
   template <typename Func> 
   void SetFunction( Func f );
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

inline Double_t TF1::operator()(Double_t x, Double_t y, Double_t z, Double_t t) const  
   { return Eval(x,y,z,t); }
inline Double_t TF1::operator()(const Double_t *x, const Double_t *params)
   { 
      if (fMethodCall) InitArgs(x,params);
      return EvalPar(x,params); 
   }


inline void TF1::SetRange(Double_t xmin, Double_t,  Double_t xmax, Double_t)
   { TF1::SetRange(xmin, xmax); }
inline void TF1::SetRange(Double_t xmin, Double_t, Double_t,  Double_t xmax, Double_t, Double_t)
   { TF1::SetRange(xmin, xmax); }

template <typename Func> 
void TF1::SetFunction( Func f )    {
   // set function from a generic C++ callable object 
   fType = 1; 
   fFunctor = ROOT::Math::ParamFunctor(f); 
} 
template <class PtrObj, typename MemFn> 
void TF1::SetFunction( PtrObj& p, MemFn memFn )   { 
   // set from a pointer to a member function
   fType = 1; 
   fFunctor = ROOT::Math::ParamFunctor(p,memFn); 
} 

#endif
