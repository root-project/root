// @(#)root/hist:$Id$
// Author: Rene Brun   18/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TMath.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraph.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TRandom.h"
#include "TInterpreter.h"
#include "TPluginManager.h"
#include "TBrowser.h"
#include "TColor.h"
#include "TClass.h"
#include "TMethodCall.h"
#include "TF1Helper.h"
#include "Math/WrappedFunction.h"
#include "Math/WrappedTF1.h"
#include "Math/BrentRootFinder.h"
#include "Math/BrentMinimizer1D.h"
#include "Math/BrentMethods.h"
#include "Math/Integrator.h"
#include "Math/GaussIntegrator.h"
#include "Math/GaussLegendreIntegrator.h"
#include "Math/AdaptiveIntegratorMultiDim.h"
#include "Math/RichardsonDerivator.h"
#include "Math/Functor.h"
#include "Fit/FitResult.h"

//#include <iostream>

Bool_t TF1::fgAbsValue    = kFALSE;
Bool_t TF1::fgRejectPoint = kFALSE;
static Double_t gErrorTF1 = 0;


ClassImp(TF1)

// class wrapping evaluation of TF1(x) - y0
class GFunc {
   const TF1* fFunction;
   const double fY0;
public:
   GFunc(const TF1* function , double y ):fFunction(function), fY0(y) {}
   double operator()(double x) const {
      return fFunction->Eval(x) - fY0;
   }
};

// class wrapping evaluation of -TF1(x)
class GInverseFunc {
   const TF1* fFunction;
public:
   GInverseFunc(const TF1* function):fFunction(function) {}
   double operator()(double x) const {
      return - fFunction->Eval(x);
   }
};

// class wrapping function evaluation directly in 1D interface (used for integration) 
// and implementing the methods for the momentum calculations

class  TF1_EvalWrapper : public ROOT::Math::IGenFunction { 
public: 
   TF1_EvalWrapper(TF1 * f, const Double_t * par, bool useAbsVal, Double_t n = 1, Double_t x0 = 0) : 
      fFunc(f), 
      fPar( ( (par) ? par : f->GetParameters() ) ),
      fAbsVal(useAbsVal),
      fN(n), 
      fX0(x0)   
   {
      fFunc->InitArgs(fX, fPar); 
   }

   ROOT::Math::IGenFunction * Clone()  const { 
      // use default copy constructor
      TF1_EvalWrapper * f =  new TF1_EvalWrapper( *this);
      f->fFunc->InitArgs(f->fX, f->fPar); 
      return f;
   }
   // evaluate |f(x)|
   Double_t DoEval( Double_t x) const { 
      fX[0] = x; 
      Double_t fval = fFunc->EvalPar( fX, fPar);
      if (fAbsVal && fval < 0)  return -fval;
      return fval; 
   } 
   // evaluate x * |f(x)|
   Double_t EvalFirstMom( Double_t x) { 
      fX[0] = x; 
      return fX[0] * TMath::Abs( fFunc->EvalPar( fX, fPar) ); 
   } 
   // evaluate (x - x0) ^n * f(x)
   Double_t EvalNMom( Double_t x) const  { 
      fX[0] = x; 
      return TMath::Power( fX[0] - fX0, fN) * TMath::Abs( fFunc->EvalPar( fX, fPar) ); 
   }

   TF1 * fFunc; 
   mutable Double_t fX[1]; 
   const double * fPar; 
   Bool_t fAbsVal;
   Double_t fN; 
   Double_t fX0;
};



//______________________________________________________________________________
/* Begin_Html
<center><h2>TF1: 1-Dim function class</h2></center>
A TF1 object is a 1-Dim function defined between a lower and upper limit.
<br>The function may be a simple function (see <tt>TFormula</tt>) or a
precompiled user function.
<br>The function may have associated parameters.
<br>TF1 graphics function is via the <tt>TH1/TGraph</tt> drawing functions.
<p>
The following types of functions can be created:
<ul>
<li><a href="#F1">A - Expression using variable x and no parameters</a></li>
<li><a href="#F2">B - Expression using variable x with parameters</a></li>
<li><a href="#F3">C - A general C function with parameters</a></li>
<li><a href="#F4">D - A general C++ function object (functor) with parameters</a></li>
<li><a href="#F5">E - A member function with parameters of a general C++ class</a></li>
</ul>

<a name="F1"></a><h3>A - Expression using variable x and no parameters</h3>
<h4>Case 1: inline expression using standard C++ functions/operators</h4>
<div class="code"><pre>
   TF1 *fa1 = new TF1("fa1","sin(x)/x",0,10);
   fa1->Draw();
</pre></div><div class="clear" />
End_Html
Begin_Macro
{
   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   TF1 *fa1 = new TF1("fa1","sin(x)/x",0,10);
   fa1->Draw();
   return c;
}
End_Macro
Begin_Html
<h4>Case 2: inline expression using TMath functions without parameters</h4>
<div class="code"><pre>
   TF1 *fa2 = new TF1("fa2","TMath::DiLog(x)",0,10);
   fa2->Draw();
</pre></div><div class="clear" />
End_Html
Begin_Macro
{
   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   TF1 *fa2 = new TF1("fa2","TMath::DiLog(x)",0,10);
   fa2->Draw();
   return c;
}
End_Macro
Begin_Html
<h4>Case 3: inline expression using a CINT function by name</h4>
<div class="code"><pre>
   Double_t myFunc(x) {
      return x+sin(x);
   }
   TF1 *fa3 = new TF1("fa3","myFunc(x)",-3,5);
   fa3->Draw();
</pre></div><div class="clear" />

<a name="F2"></a><h3>B - Expression using variable x with parameters</h3>
<h4>Case 1: inline expression using standard C++ functions/operators</h4>
<ul>
<li>Example a:
<div class="code"><pre>
   TF1 *fa = new TF1("fa","[0]*x*sin([1]*x)",-3,3);
</pre></div><div class="clear" />
This creates a function of variable x with 2 parameters.
The parameters must be initialized via:
<pre>
   fa->SetParameter(0,value_first_parameter);
   fa->SetParameter(1,value_second_parameter);
</pre>
Parameters may be given a name:
<pre>
   fa->SetParName(0,"Constant");
</pre>
</li>
<li> Example b:
<div class="code"><pre>
   TF1 *fb = new TF1("fb","gaus(0)*expo(3)",0,10);
</pre></div><div class="clear" />
<tt>gaus(0)</tt> is a substitute for <tt>[0]*exp(-0.5*((x-[1])/[2])**2)</tt>
and <tt>(0)</tt> means start numbering parameters at <tt>0</tt>.
<tt>expo(3)</tt> is a substitute for <tt>exp([3]+[4]*x)</tt>.
</li>
</ul>

<h4>Case 2: inline expression using TMath functions with parameters</h4>
<div class="code"><pre>
   TF1 *fb2 = new TF1("fa3","TMath::Landau(x,[0],[1],0)",-5,10);
   fb2->SetParameters(0.2,1.3);
   fb2->Draw();
</pre></div><div class="clear" />
End_Html
Begin_Macro
{
   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   TF1 *fb2 = new TF1("fa3","TMath::Landau(x,[0],[1],0)",-5,10);
   fb2->SetParameters(0.2,1.3);
   fb2->Draw();
   return c;
}
End_Macro
Begin_Html

<a name="F3"></a><h3>C - A general C function with parameters</h3>
Consider the macro myfunc.C below:
<div class="code"><pre>
   // Macro myfunc.C
   Double_t myfunction(Double_t *x, Double_t *par)
   {
      Float_t xx =x[0];
      Double_t f = TMath::Abs(par[0]*sin(par[1]*xx)/xx);
      return f;
   }
   void myfunc()
   {
      TF1 *f1 = new TF1("myfunc",myfunction,0,10,2);
      f1->SetParameters(2,1);
      f1->SetParNames("constant","coefficient");
      f1->Draw();
   }
   void myfit()
   {
      TH1F *h1=new TH1F("h1","test",100,0,10);
      h1->FillRandom("myfunc",20000);
      TF1 *f1=gROOT->GetFunction("myfunc");
      f1->SetParameters(800,1);
      h1->Fit("myfunc");
   }
</pre></div><div class="clear" />

End_Html
Begin_Html

<p>
In an interactive session you can do:
<div class="code"><pre>
   Root > .L myfunc.C
   Root > myfunc();
   Root > myfit();
</pre></div>
<div class="clear" />

End_Html
Begin_Html

<tt>TF1</tt> objects can reference other <tt>TF1</tt> objects (thanks John
Odonnell) of type A or B defined above. This excludes CINT interpreted functions
and compiled functions. However, there is a restriction. A function cannot
reference a basic function if the basic function is a polynomial polN.
<p>Example:
<div class="code"><pre>
   {
      TF1 *fcos = new TF1 ("fcos", "[0]*cos(x)", 0., 10.);
      fcos->SetParNames( "cos");
      fcos->SetParameter( 0, 1.1);

      TF1 *fsin = new TF1 ("fsin", "[0]*sin(x)", 0., 10.);
      fsin->SetParNames( "sin");
      fsin->SetParameter( 0, 2.1);

      TF1 *fsincos = new TF1 ("fsc", "fcos+fsin");

      TF1 *fs2 = new TF1 ("fs2", "fsc+fsc");
   }
</pre></div><div class="clear" />

End_Html
Begin_Html


<a name="F4"></a><h3>D - A general C++ function object (functor) with parameters</h3>
A TF1 can be created from any C++ class implementing the operator()(double *x, double *p).
The advantage of the function object is that he can have a state and reference therefore what-ever other object.
In this way the user can customize his function.
<p>Example:
<div class="code"><pre>
class  MyFunctionObject {
 public:
   // use constructor to customize your function object

   double operator() (double *x, double *p) {
      // function implementation using class data members
   }
};
{
    ....
   MyFunctionObject * fobj = new MyFunctionObject(....);       // create the function object
   TF1 * f = new TF1("f",fobj,0,1,npar,"MyFunctionObject");    // create TF1 class.
   .....
}
</pre></div><div class="clear" />
When constructing the TF1 class, the name of the function object class is required only if running in CINT
and it is not needed in compiled C++ mode. In addition in compiled mode the cfnution object can be passed to TF1
by value.
See also the tutorial math/exampleFunctor.C for a running example.

End_Html
Begin_Html

<a name="F5"></a><h3>E - A member function with parameters of a general C++ class</h3>
A TF1 can be created in this case from any member function of a class which has the signature of
(double * , double *) and returning a double.
<p>Example:
<div class="code"><pre>
class  MyFunction {
 public:
   ...
   double Evaluate() (double *x, double *p) {
      // function implementation
   }
};
{
    ....
   MyFunction * fptr = new MyFunction(....);  // create the user function class
   TF1 * f = new TF1("f",fptr,&MyFunction::Evaluate,0,1,npar,"MyFunction","Evaluate");   // create TF1 class.

   .....
}
</pre></div><div class="clear" />
When constructing the TF1 class, the name of the function class and of the member function are required only
if running in CINT and they are not need in compiled C++ mode.
See also the tutorial math/exampleFunctor.C for a running example.

End_Html */

TF1 *TF1::fgCurrent = 0;


//______________________________________________________________________________
TF1::TF1(): TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 default constructor.

   fXmin      = 0;
   fXmax      = 0;
   fNpx       = 100;
   fType      = 0;
   fNpfits    = 0;
   fNDF       = 0;
   fNsave     = 0;
   fChisquare = 0;
   fIntegral  = 0;
   fParErrors = 0;
   fParMin    = 0;
   fParMax    = 0;
   fAlpha     = 0;
   fBeta      = 0;
   fGamma     = 0;
   fParent    = 0;
   fSave      = 0;
   fHistogram = 0;
   fMinimum   = -1111;
   fMaximum   = -1111;
   fMethodCall = 0;
   fCintFunc   = 0;
   SetFillStyle(0);
}


//______________________________________________________________________________
TF1::TF1(const char *name,const char *formula, Double_t xmin, Double_t xmax)
      :TFormula(name,formula), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor using a formula definition
   //
   //  See TFormula constructor for explanation of the formula syntax.
   //
   //  See tutorials: fillrandom, first, fit1, formula1, multifit
   //  for real examples.
   //
   //  Creates a function of type A or B between xmin and xmax
   //
   //  if formula has the form "fffffff;xxxx;yyyy", it is assumed that
   //  the formula string is "fffffff" and "xxxx" and "yyyy" are the
   //  titles for the X and Y axis respectively.

   if (xmin < xmax ) {
      fXmin      = xmin;
      fXmax      = xmax;
   } else {
      fXmin = xmax; //when called from TF2,TF3
      fXmax = xmin;
   }
   fNpx       = 100;
   fType      = 0;
   if (fNpar) {
      fParErrors = new Double_t[fNpar];
      fParMin    = new Double_t[fNpar];
      fParMax    = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fNsave      = 0;
   fSave       = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fMethodCall = 0;
   fCintFunc   = 0;

   if (fNdim != 1 && xmin < xmax) {
      Error("TF1","function: %s/%s has %d parameters instead of 1",name,formula,fNdim);
      MakeZombie();
   }

   if (!gStyle) return;
   SetLineColor(gStyle->GetFuncColor());
   SetLineWidth(gStyle->GetFuncWidth());
   SetLineStyle(gStyle->GetFuncStyle());
   SetFillStyle(0);
}


//______________________________________________________________________________
TF1::TF1(const char *name, Double_t xmin, Double_t xmax, Int_t npar)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor using name of an interpreted function.
   //
   //  Creates a function of type C between xmin and xmax.
   //  name is the name of an interpreted CINT cunction.
   //  The function is defined with npar parameters
   //  fcn must be a function of type:
   //     Double_t fcn(Double_t *x, Double_t *params)
   //
   //  This constructor is called for functions of type C by CINT.
   //
   // WARNING! A function created with this constructor cannot be Cloned.

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;
   fType       = 2;
   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fNsave      = 0;
   fSave       = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fMethodCall = 0;
   fCintFunc   = 0;
   fNdim       = 1;

   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);

   if (gStyle) {
      SetLineColor(gStyle->GetFuncColor());
      SetLineWidth(gStyle->GetFuncWidth());
      SetLineStyle(gStyle->GetFuncStyle());
   }
   SetFillStyle(0);

   SetTitle(name);
   if (name) {
      if (*name == '*') return; //case happens via SavePrimitive
      fMethodCall = new TMethodCall();
      fMethodCall->InitWithPrototype(name,"Double_t*,Double_t*");
      fNumber = -1;
      gROOT->GetListOfFunctions()->Add(this);
      if (! fMethodCall->IsValid() ) {
         Error("TF1","No function found with the signature %s(Double_t*,Double_t*)",name);
      }
   } else {
      Error("TF1","requires a proper function name!");
   }
}


//______________________________________________________________________________
TF1::TF1(const char *name,void *fcn, Double_t xmin, Double_t xmax, Int_t npar)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor using pointer to an interpreted function.
   //
   //  See TFormula constructor for explanation of the formula syntax.
   //
   //  Creates a function of type C between xmin and xmax.
   //  The function is defined with npar parameters
   //  fcn must be a function of type:
   //     Double_t fcn(Double_t *x, Double_t *params)
   //
   //  see tutorial; myfit for an example of use
   //  also test/stress.cxx (see function stress1)
   //
   //
   //  This constructor is called for functions of type C by CINT.
   //
   //  WARNING! A function created with this constructor cannot be Cloned.


   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;
   fType       = 2;
   //fFunction   = 0;
   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fNsave      = 0;
   fSave       = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fMethodCall = 0;
   fCintFunc   = 0;
   fNdim       = 1;

   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);

   if (gStyle) {
      SetLineColor(gStyle->GetFuncColor());
      SetLineWidth(gStyle->GetFuncWidth());
      SetLineStyle(gStyle->GetFuncStyle());
   }
   SetFillStyle(0);

   if (!fcn) return;
   const char *funcname = gCint->Getp2f2funcname(fcn);
   SetTitle(funcname);
   if (funcname) {
      fMethodCall = new TMethodCall();
      fMethodCall->InitWithPrototype(funcname,"Double_t*,Double_t*");
      fNumber = -1;
      gROOT->GetListOfFunctions()->Add(this);
      if (! fMethodCall->IsValid() ) {
         Error("TF1","No function found with the signature %s(Double_t*,Double_t*)",funcname);
      }
   } else {
      Error("TF1","can not find any function at the address 0x%lx. This function requested for %s",(Long_t)fcn,name);
   }


}


//______________________________________________________________________________
TF1::TF1(const char *name,Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin, Double_t xmax, Int_t npar)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor using a pointer to a real function.
   //
   //   npar is the number of free parameters used by the function
   //
   //   This constructor creates a function of type C when invoked
   //   with the normal C++ compiler.
   //
   //   see test program test/stress.cxx (function stress1) for an example.
   //   note the interface with an intermediate pointer.
   //
   // WARNING! A function created with this constructor cannot be Cloned.

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;

   fType       = 1;
   fMethodCall = 0;
   fCintFunc   = 0;
   fFunctor = ROOT::Math::ParamFunctor(fcn);

   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fNsave      = 0;
   fSave       = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fNdim       = 1;

   // Store formula in linked list of formula in ROOT
   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);
   gROOT->GetListOfFunctions()->Add(this);

   if (!gStyle) return;
   SetLineColor(gStyle->GetFuncColor());
   SetLineWidth(gStyle->GetFuncWidth());
   SetLineStyle(gStyle->GetFuncStyle());
   SetFillStyle(0);

}

//______________________________________________________________________________
TF1::TF1(const char *name,Double_t (*fcn)(const Double_t *, const Double_t *), Double_t xmin, Double_t xmax, Int_t npar)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor using a pointer to real function.
   //
   //   npar is the number of free parameters used by the function
   //
   //   This constructor creates a function of type C when invoked
   //   with the normal C++ compiler.
   //
   //   see test program test/stress.cxx (function stress1) for an example.
   //   note the interface with an intermediate pointer.
   //
   // WARNING! A function created with this constructor cannot be Cloned.

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;

   fType       = 1;
   fMethodCall = 0;
   fCintFunc   = 0;
   fFunctor = ROOT::Math::ParamFunctor(fcn);

   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fNsave      = 0;
   fSave       = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fNdim       = 1;

   // Store formula in linked list of formula in ROOT
   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);
   gROOT->GetListOfFunctions()->Add(this);

   if (!gStyle) return;
   SetLineColor(gStyle->GetFuncColor());
   SetLineWidth(gStyle->GetFuncWidth());
   SetLineStyle(gStyle->GetFuncStyle());
   SetFillStyle(0);

}


//______________________________________________________________________________
TF1::TF1(const char *name, ROOT::Math::ParamFunctor f, Double_t xmin, Double_t xmax, Int_t npar ) :
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
   fFunctor   ( ROOT::Math::ParamFunctor(f) )
{
   // F1 constructor using the Functor class.
   //
   //   xmin and xmax define the plotting range of the function
   //   npar is the number of free parameters used by the function
   //
   //   This constructor can be used only in compiled code
   //
   // WARNING! A function created with this constructor cannot be Cloned.

   CreateFromFunctor(name, npar);
}


//______________________________________________________________________________
void TF1::CreateFromFunctor(const char *name, Int_t npar)
{
   // Internal Function to Create a TF1  using a Functor.
   //
   //          Used by the template constructors

   fNdim       = 1;

   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }

   // Store formula in linked list of formula in ROOT
   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);
   gROOT->GetListOfFunctions()->Add(this);

   if (!gStyle) return;
   SetLineColor(gStyle->GetFuncColor());
   SetLineWidth(gStyle->GetFuncWidth());
   SetLineStyle(gStyle->GetFuncStyle());
   SetFillStyle(0);

}

//______________________________________________________________________________
TF1::TF1(const char *name,void *ptr, Double_t xmin, Double_t xmax, Int_t npar, const char * className )
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor from an interpreted class defining the operator() or Eval().
   // This constructor emulate the syntax of the template constructor using a C++ callable object (functor)
   // which can be used only in C++ compiled mode.
   // The class name is required to get the type of class given the void pointer ptr.
   // For the method name is used the operator() (double *, double * ).
   // Use the other constructor taking the method name for different method names.
   //
   //  xmin and xmax specify the function plotting range
   //  npar are the number of function parameters
   //
   //  see tutorial  math.exampleFunctor.C for an example of using this constructor
   //
   //  This constructor is used only when using CINT.
   //  In compiled mode the template constructor is used and in that case className is not needed

   CreateFromCintClass(name, ptr, xmin, xmax, npar, className, 0 );
}

//______________________________________________________________________________
TF1::TF1(const char *name,void *ptr, void * , Double_t xmin, Double_t xmax, Int_t npar, const char * className, const char * methodName)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
   // F1 constructor from an interpreter class using a specidied member function.
   // This constructor emulate the syntax of the template constructor using a C++ class and a given
   // member function pointer, which can be used only in C++ compiled mode.
   // The class name is required to get the type of class given the void pointer ptr.
   // The second void * is not needed for the CINT case, but is kept for emulating the API of the
   // template constructor.
   // The method name is optional. By default is looked for operator() (double *, double *) or
   // Eval(double *, double*)
   //
   //  xmin and xmax specify the function plotting range
   //  npar are the number of function parameters.
   //
   //
   //  see tutorial  math.exampleFunctor.C for an example of using this constructor
   //
   //  This constructor is used only when using CINT.
   //  In compiled mode the template constructor is used and in that case className is not needed

   CreateFromCintClass(name, ptr, xmin, xmax, npar, className, methodName);
}

//______________________________________________________________________________
void TF1::CreateFromCintClass(const char *name,void *ptr, Double_t xmin, Double_t xmax, Int_t npar, const char * className, const char * methodName)
{
   // Internal function used to create from TF1 from an interpreter CINT class
   // with the specified type (className) and member function name (methodName).
   //


   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;
   fType       = 3;
   if (npar > 0 ) fNpar = npar;
   if (fNpar) {
      fNames      = new TString[fNpar];
      fParams     = new Double_t[fNpar];
      fParErrors  = new Double_t[fNpar];
      fParMin     = new Double_t[fNpar];
      fParMax     = new Double_t[fNpar];
      for (int i = 0; i < fNpar; i++) {
         fParams[i]     = 0;
         fParErrors[i]  = 0;
         fParMin[i]     = 0;
         fParMax[i]     = 0;
      }
   } else {
      fParErrors = 0;
      fParMin    = 0;
      fParMax    = 0;
   }
   fChisquare  = 0;
   fIntegral   = 0;
   fAlpha      = 0;
   fBeta       = 0;
   fGamma      = 0;
   fParent     = 0;
   fNpfits     = 0;
   fNDF        = 0;
   fNsave      = 0;
   fSave       = 0;
   fHistogram  = 0;
   fMinimum    = -1111;
   fMaximum    = -1111;
   fMethodCall = 0;
   fNdim       = 1;

   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   gROOT->GetListOfFunctions()->Remove(f1old);
   SetName(name);

   if (gStyle) {
      SetLineColor(gStyle->GetFuncColor());
      SetLineWidth(gStyle->GetFuncWidth());
      SetLineStyle(gStyle->GetFuncStyle());
   }
   SetFillStyle(0);

   if (!ptr) return;
   fCintFunc = ptr;

   if (!className) return;

   TClass *cl = TClass::GetClass(className);

   if (cl) {
      fMethodCall = new TMethodCall();


      if (methodName)
         fMethodCall->InitWithPrototype(cl,methodName,"Double_t*,Double_t*");
      else {
         fMethodCall->InitWithPrototype(cl,"operator()","Double_t*,Double_t*");
         if (! fMethodCall->IsValid() )
            // try with Eval if operator() is not found
            fMethodCall->InitWithPrototype(cl,"Eval","Double_t*,Double_t*");
      }

      fNumber = -1;
      gROOT->GetListOfFunctions()->Add(this);
      if (! fMethodCall->IsValid() ) {
         if (methodName)
            Error("TF1","No function found in class %s with the signature %s(Double_t*,Double_t*)",className,methodName);
         else
            Error("TF1","No function found in class %s with the signature operator() (Double_t*,Double_t*) or Eval(Double_t*,Double_t*)",className);
      }
   } else {
      Error("TF1","can not find any class with name %s at the address 0x%lx",className,(Long_t)ptr);
   }


}



//______________________________________________________________________________
TF1& TF1::operator=(const TF1 &rhs)
{
   // Operator =

   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}


//______________________________________________________________________________
TF1::~TF1()
{
   // TF1 default destructor.

   if (fParMin)    delete [] fParMin;
   if (fParMax)    delete [] fParMax;
   if (fParErrors) delete [] fParErrors;
   if (fIntegral)  delete [] fIntegral;
   if (fAlpha)     delete [] fAlpha;
   if (fBeta)      delete [] fBeta;
   if (fGamma)     delete [] fGamma;
   if (fSave)      delete [] fSave;
   delete fHistogram;
   delete fMethodCall;

   if (fParent) fParent->RecursiveRemove(this);
}


//______________________________________________________________________________
TF1::TF1(const TF1 &f1) : TFormula(), TAttLine(f1), TAttFill(f1), TAttMarker(f1)
{
   // Constuctor.

   fXmin      = 0;
   fXmax      = 0;
   fNpx       = 100;
   fType      = 0;
   fNpfits    = 0;
   fNDF       = 0;
   fNsave     = 0;
   fChisquare = 0;
   fIntegral  = 0;
   fParErrors = 0;
   fParMin    = 0;
   fParMax    = 0;
   fAlpha     = 0;
   fBeta      = 0;
   fGamma     = 0;
   fParent    = 0;
   fSave      = 0;
   fHistogram = 0;
   fMinimum   = -1111;
   fMaximum   = -1111;
   fMethodCall = 0;
   fCintFunc   = 0;
   SetFillStyle(0);

   ((TF1&)f1).Copy(*this);
}


//______________________________________________________________________________
void TF1::AbsValue(Bool_t flag)
{
   // Static function: set the fgAbsValue flag.
   // By default TF1::Integral uses the original function value to compute the integral
   // However, TF1::Moment, CentralMoment require to compute the integral
   // using the absolute value of the function.

   fgAbsValue = flag;
}


//______________________________________________________________________________
void TF1::Browse(TBrowser *b)
{
   // Browse.

   Draw(b ? b->GetDrawOption() : "");
   gPad->Update();
}


//______________________________________________________________________________
void TF1::Copy(TObject &obj) const
{
   // Copy this F1 to a new F1.

   if (((TF1&)obj).fParMin)    delete [] ((TF1&)obj).fParMin;
   if (((TF1&)obj).fParMax)    delete [] ((TF1&)obj).fParMax;
   if (((TF1&)obj).fParErrors) delete [] ((TF1&)obj).fParErrors;
   if (((TF1&)obj).fIntegral)  delete [] ((TF1&)obj).fIntegral;
   if (((TF1&)obj).fAlpha)     delete [] ((TF1&)obj).fAlpha;
   if (((TF1&)obj).fBeta)      delete [] ((TF1&)obj).fBeta;
   if (((TF1&)obj).fGamma)     delete [] ((TF1&)obj).fGamma;
   if (((TF1&)obj).fSave)      delete [] ((TF1&)obj).fSave;
   delete ((TF1&)obj).fHistogram;
   delete ((TF1&)obj).fMethodCall;

   TFormula::Copy(obj);
   TAttLine::Copy((TF1&)obj);
   TAttFill::Copy((TF1&)obj);
   TAttMarker::Copy((TF1&)obj);
   ((TF1&)obj).fXmin = fXmin;
   ((TF1&)obj).fXmax = fXmax;
   ((TF1&)obj).fNpx  = fNpx;
   ((TF1&)obj).fType = fType;
   ((TF1&)obj).fCintFunc  = fCintFunc;
   ((TF1&)obj).fFunctor   = fFunctor;
   ((TF1&)obj).fChisquare = fChisquare;
   ((TF1&)obj).fNpfits  = fNpfits;
   ((TF1&)obj).fNDF     = fNDF;
   ((TF1&)obj).fMinimum = fMinimum;
   ((TF1&)obj).fMaximum = fMaximum;

   ((TF1&)obj).fParErrors = 0;
   ((TF1&)obj).fParMin    = 0;
   ((TF1&)obj).fParMax    = 0;
   ((TF1&)obj).fIntegral  = 0;
   ((TF1&)obj).fAlpha     = 0;
   ((TF1&)obj).fBeta      = 0;
   ((TF1&)obj).fGamma     = 0;
   ((TF1&)obj).fParent    = fParent;
   ((TF1&)obj).fNsave     = fNsave;
   ((TF1&)obj).fSave      = 0;
   ((TF1&)obj).fHistogram = 0;
   ((TF1&)obj).fMethodCall = 0;
   if (fNsave) {
      ((TF1&)obj).fSave = new Double_t[fNsave];
      for (Int_t j=0;j<fNsave;j++) ((TF1&)obj).fSave[j] = fSave[j];
   }
   if (fNpar) {
      ((TF1&)obj).fParErrors = new Double_t[fNpar];
      ((TF1&)obj).fParMin    = new Double_t[fNpar];
      ((TF1&)obj).fParMax    = new Double_t[fNpar];
      Int_t i;
      for (i=0;i<fNpar;i++)   ((TF1&)obj).fParErrors[i] = fParErrors[i];
      for (i=0;i<fNpar;i++)   ((TF1&)obj).fParMin[i]    = fParMin[i];
      for (i=0;i<fNpar;i++)   ((TF1&)obj).fParMax[i]    = fParMax[i];
   }
   if (fMethodCall) {
      // use copy-constructor of TMethodCall 
      TMethodCall *m = new TMethodCall(*fMethodCall);
//       m->InitWithPrototype(fMethodCall->GetMethodName(),fMethodCall->GetProto());
      ((TF1&)obj).fMethodCall  = m;
   }
}


//______________________________________________________________________________
Double_t TF1::Derivative(Double_t x, Double_t *params, Double_t eps) const
{
   // Returns the first derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   // D(h) = #frac{f(x+h) - f(x-h)}{2h}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // if the argument params is null, the current function parameters are used,
   // otherwise the parameters in params are used.
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.
   //
   // Author: Anna Kreshuk
   
   if (GetNdim() > 1) { 
      Warning("Derivative","Function dimension is larger than one");
   }

   ROOT::Math::RichardsonDerivator rd;
   double xmin, xmax;
   GetRange(xmin, xmax);
   // this is not optimal (should be used the average x instead of the range) 
   double h = eps* std::abs(xmax-xmin);
   if ( h <= 0 ) h = 0.001;  
   double der = 0; 
   if (params) { 
      ROOT::Math::WrappedTF1 wtf(*( const_cast<TF1 *> (this) )); 
      wtf.SetParameters(params);
      der = rd.Derivative1(wtf,x,h);   
   }                                            
   else { 
      // no need to set parameters used a non-parametric wrapper to avoid allocating 
      // an array with parameter values
      ROOT::Math::WrappedFunction<const TF1 & > wf( *this);
      der = rd.Derivative1(wf,x,h);   
   }

   gErrorTF1 = rd.Error();
   return der;

}


//______________________________________________________________________________
Double_t TF1::Derivative2(Double_t x, Double_t *params, Double_t eps) const
{
   // Returns the second derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   //    D(h) = #frac{f(x+h) - 2f(x) + f(x-h)}{h^{2}}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // if the argument params is null, the current function parameters are used,
   // otherwise the parameters in params are used.
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.
   //
   // Author: Anna Kreshuk

   if (GetNdim() > 1) { 
      Warning("Derivative2","Function dimension is larger than one");
   }

   ROOT::Math::RichardsonDerivator rd;
   double xmin, xmax;
   GetRange(xmin, xmax);
   // this is not optimal (should be used the average x instead of the range) 
   double h = eps* std::abs(xmax-xmin);
   if ( h <= 0 ) h = 0.001;  
   double der = 0; 
   if (params) { 
      ROOT::Math::WrappedTF1 wtf(*( const_cast<TF1 *> (this) )); 
      wtf.SetParameters(params);
      der = rd.Derivative2(wtf,x,h);   
   }                                            
   else { 
      // no need to set parameters used a non-parametric wrapper to avoid allocating 
      // an array with parameter values
      ROOT::Math::WrappedFunction<const TF1 & > wf( *this);
      der = rd.Derivative2(wf,x,h);   
   }

   gErrorTF1 = rd.Error();

   return der;
}


//______________________________________________________________________________
Double_t TF1::Derivative3(Double_t x, Double_t *params, Double_t eps) const
{
   // Returns the third derivative of the function at point x,
   // computed by Richardson's extrapolation method (use 2 derivative estimates
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //Begin_Latex
   //    D(h) = #frac{f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)}{2h^{3}}
   //End_Latex
   // the final estimate Begin_Latex D = #frac{4D(h/2) - D(h)}{3} End_Latex
   //  "Numerical Methods for Scientists and Engineers", H.M.Antia, 2nd edition"
   //
   // if the argument params is null, the current function parameters are used,
   // otherwise the parameters in params are used.
   //
   // the argument eps may be specified to control the step size (precision).
   // the step size is taken as eps*(xmax-xmin).
   // the default value (0.001) should be good enough for the vast majority
   // of functions. Give a smaller value if your function has many changes
   // of the second derivative in the function range.
   //
   // Getting the error via TF1::DerivativeError:
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //Begin_Latex
   //    err = k#sqrt{f(x)^{2} + x^{2}deriv^{2}}#sqrt{#sum ai^{2}},
   //End_Latex
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.
   //
   // Author: Anna Kreshuk

   if (GetNdim() > 1) { 
      Warning("Derivative3","Function dimension is larger than one");
   }

   ROOT::Math::RichardsonDerivator rd;
   double xmin, xmax;
   GetRange(xmin, xmax);
   // this is not optimal (should be used the average x instead of the range) 
   double h = eps* std::abs(xmax-xmin);
   if ( h <= 0 ) h = 0.001;  
   double der = 0; 
   if (params) { 
      ROOT::Math::WrappedTF1 wtf(*( const_cast<TF1 *> (this) )); 
      wtf.SetParameters(params);
      der = rd.Derivative3(wtf,x,h);   
   }                                            
   else { 
      // no need to set parameters used a non-parametric wrapper to avoid allocating 
      // an array with parameter values
      ROOT::Math::WrappedFunction<const TF1 & > wf( *this);
      der = rd.Derivative3(wf,x,h);   
   }

   gErrorTF1 = rd.Error();
   return der; 

}


//______________________________________________________________________________
Double_t TF1::DerivativeError()
{
   // Static function returning the error of the last call to the of Derivative's
   // functions

   return gErrorTF1;
}


//______________________________________________________________________________
Int_t TF1::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a function.
   //
   //  Compute the closest distance of approach from point px,py to this
   //  function. The distance is computed in pixels units.
   //
   //  Note that px is called with a negative value when the TF1 is in
   //  TGraph or TH1 list of functions. In this case there is no point
   //  looking at the histogram axis.

   if (!fHistogram) return 9999;
   Int_t distance = 9999;
   if (px >= 0) {
      distance = fHistogram->DistancetoPrimitive(px,py);
      if (distance <= 1) return distance;
   } else {
      px = -px;
   }

   Double_t xx[1];
   Double_t x    = gPad->AbsPixeltoX(px);
   xx[0]         = gPad->PadtoX(x);
   if (xx[0] < fXmin || xx[0] > fXmax) return distance;
   Double_t fval = Eval(xx[0]);
   Double_t y    = gPad->YtoPad(fval);
   Int_t pybin   = gPad->YtoAbsPixel(y);
   return TMath::Abs(py - pybin);
}


//______________________________________________________________________________
void TF1::Draw(Option_t *option)
{
   // Draw this function with its current attributes.
   //
   // Possible option values are:
   //   "SAME"  superimpose on top of existing picture
   //   "L"     connect all computed points with a straight line
   //   "C"     connect all computed points with a smooth curve
   //   "FC"    draw a fill area below a smooth curve
   //
   // Note that the default value is "L". Therefore to draw on top
   // of an existing picture, specify option "LSAME"
   //
   // NB. You must use DrawCopy if you want to draw several times the same
   //     function in the current canvas.

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();

   AppendPad(option);
}


//______________________________________________________________________________
TF1 *TF1::DrawCopy(Option_t *option) const
{
   // Draw a copy of this function with its current attributes.
   //
   //  This function MUST be used instead of Draw when you want to draw
   //  the same function with different parameters settings in the same canvas.
   //
   // Possible option values are:
   //   "SAME"  superimpose on top of existing picture
   //   "L"     connect all computed points with a straight line
   //   "C"     connect all computed points with a smooth curve
   //   "FC"    draw a fill area below a smooth curve
   //
   // Note that the default value is "L". Therefore to draw on top
   // of an existing picture, specify option "LSAME"

   TF1 *newf1 = (TF1*)this->IsA()->New();
   Copy(*newf1);
   newf1->AppendPad(option);
   newf1->SetBit(kCanDelete);
   return newf1;
}


//______________________________________________________________________________
TObject *TF1::DrawDerivative(Option_t *option)
{
   // Draw derivative of this function
   //
   // An intermediate TGraph object is built and drawn with option.
   // The function returns a pointer to the TGraph object. Do:
   //    TGraph *g = (TGraph*)myfunc.DrawDerivative(option);
   //
   // The resulting graph will be drawn into the current pad.
   // If this function is used via the context menu, it recommended
   // to create a new canvas/pad before invoking this function.

   TVirtualPad *pad = gROOT->GetSelectedPad();
   TVirtualPad *padsav = gPad;
   if (pad) pad->cd();

   TGraph *gr = new TGraph(this,"d");
   gr->Draw(option);
   if (padsav) padsav->cd();
   return gr;
}


//______________________________________________________________________________
TObject *TF1::DrawIntegral(Option_t *option)
{
   // Draw integral of this function
   //
   // An intermediate TGraph object is built and drawn with option.
   // The function returns a pointer to the TGraph object. Do:
   //    TGraph *g = (TGraph*)myfunc.DrawIntegral(option);
   //
   // The resulting graph will be drawn into the current pad.
   // If this function is used via the context menu, it recommended
   // to create a new canvas/pad before invoking this function.

   TVirtualPad *pad = gROOT->GetSelectedPad();
   TVirtualPad *padsav = gPad;
   if (pad) pad->cd();

   TGraph *gr = new TGraph(this,"i");
   gr->Draw(option);
   if (padsav) padsav->cd();
   return gr;
}


//______________________________________________________________________________
void TF1::DrawF1(const char *formula, Double_t xmin, Double_t xmax, Option_t *option)
{
   // Draw formula between xmin and xmax.

   if (Compile(formula)) return;

   SetRange(xmin, xmax);

   Draw(option);
}


//______________________________________________________________________________
Double_t TF1::Eval(Double_t x, Double_t y, Double_t z, Double_t t) const
{
   // Evaluate this formula.
   //
   //   Computes the value of this function (general case for a 3-d function)
   //   at point x,y,z.
   //   For a 1-d function give y=0 and z=0
   //   The current value of variables x,y,z is passed through x, y and z.
   //   The parameters used will be the ones in the array params if params is given
   //    otherwise parameters will be taken from the stored data members fParams

   Double_t xx[4];
   xx[0] = x;
   xx[1] = y;
   xx[2] = z;
   xx[3] = t;

   ((TF1*)this)->InitArgs(xx,fParams);

   return ((TF1*)this)->EvalPar(xx,fParams);
}


//______________________________________________________________________________
Double_t TF1::EvalPar(const Double_t *x, const Double_t *params)
{
   // Evaluate function with given coordinates and parameters.
   //
   // Compute the value of this function at point defined by array x
   // and current values of parameters in array params.
   // If argument params is omitted or equal 0, the internal values
   // of parameters (array fParams) will be used instead.
   // For a 1-D function only x[0] must be given.
   // In case of a multi-dimemsional function, the arrays x must be
   // filled with the corresponding number of dimensions.
   //
   // WARNING. In case of an interpreted function (fType=2), it is the
   // user's responsability to initialize the parameters via InitArgs
   // before calling this function.
   // InitArgs should be called at least once to specify the addresses
   // of the arguments x and params.
   // InitArgs should be called everytime these addresses change.

   fgCurrent = this;

   if (fType == 0) return TFormula::EvalPar(x,params);
   Double_t result = 0;
   if (fType == 1)  {
//       if (fFunction) {
//          if (params) result = (*fFunction)((Double_t*)x,(Double_t*)params);
//          else        result = (*fFunction)((Double_t*)x,fParams);
      if (!fFunctor.Empty()) {
         if (params) result = fFunctor((Double_t*)x,(Double_t*)params);
         else        result = fFunctor((Double_t*)x,fParams);

      }else          result = GetSave(x);
      return result;
   }
   if (fType == 2) {
      if (fMethodCall) fMethodCall->Execute(result);
      else             result = GetSave(x);
      return result;
   }
   if (fType == 3) {
      //std::cout << "Eval interp function object  " << fCintFunc << " result = " << result << std::endl;
      if (fMethodCall) fMethodCall->Execute(fCintFunc,result);
      else             result = GetSave(x);
      return result;
   }
   return result;
}


//______________________________________________________________________________
void TF1::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   // Execute action corresponding to one event.
   //
   //  This member function is called when a F1 is clicked with the locator

   if (fHistogram) fHistogram->ExecuteEvent(event,px,py);

   if (!gPad->GetView()) {
      if (event == kMouseMotion)  gPad->SetCursor(kHand);
   }
}


//______________________________________________________________________________
void TF1::FixParameter(Int_t ipar, Double_t value)
{
   // Fix the value of a parameter
   // The specified value will be used in a fit operation

   if (ipar < 0 || ipar > fNpar-1) return;
   SetParameter(ipar,value);
   if (value != 0) SetParLimits(ipar,value,value);
   else            SetParLimits(ipar,1,1);
}


//______________________________________________________________________________
TF1 *TF1::GetCurrent()
{
   // Static function returning the current function being processed

   return fgCurrent;
}


//______________________________________________________________________________
TH1 *TF1::GetHistogram() const
{
   // Return a pointer to the histogram used to vusualize the function

   if (fHistogram) return fHistogram;

   // May be function has not yet be painted. force a pad update
   ((TF1*)this)->Paint();
   return fHistogram;
}


//______________________________________________________________________________
Double_t TF1::GetMaximum(Double_t xmin, Double_t xmax, Double_t epsilon, Int_t maxiter,Bool_t logx) const
{
   // Return the maximum value of the function
   // Method:
   //  First, the grid search is used to bracket the maximum
   //  with the step size = (xmax-xmin)/fNpx.
   //  This way, the step size can be controlled via the SetNpx() function.
   //  If the function is unimodal or if its extrema are far apart, setting
   //  the fNpx to a small value speeds the algorithm up many times.
   //  Then, Brent's method is applied on the bracketed interval
   //  epsilon (default = 1.E-10) controls the relative accuracy (if |x| > 1 ) 
   //  and absolute (if |x| < 1)  and maxiter (default = 100) controls the maximum number 
   //  of iteration of the Brent algorithm
   //  If the flag logx is set the grid search is done in log step size
   //  This is done automatically if the log scale is set in the current Pad
   //
   // NOTE: see also TF1::GetMaximumX and TF1::GetX

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}

   if (!logx && gPad != 0) logx = gPad->GetLogx(); 

   ROOT::Math::BrentMinimizer1D bm;
   GInverseFunc g(this);
   ROOT::Math::WrappedFunction<GInverseFunc> wf1(g);
   bm.SetFunction( wf1, xmin, xmax );
   bm.SetNpx(fNpx);
   bm.SetLogScan(logx);
   bm.Minimize(maxiter, epsilon, epsilon );
   Double_t x;
   x = - bm.FValMinimum();

   return x;
}


//______________________________________________________________________________
Double_t TF1::GetMaximumX(Double_t xmin, Double_t xmax, Double_t epsilon, Int_t maxiter,Bool_t logx) const
{
   // Return the X value corresponding to the maximum value of the function
   // Method:
   //  First, the grid search is used to bracket the maximum
   //  with the step size = (xmax-xmin)/fNpx.
   //  This way, the step size can be controlled via the SetNpx() function.
   //  If the function is unimodal or if its extrema are far apart, setting
   //  the fNpx to a small value speeds the algorithm up many times.
   //  Then, Brent's method is applied on the bracketed interval
   //  epsilon (default = 1.E-10) controls the relative accuracy (if |x| > 1 ) 
   //  and absolute (if |x| < 1)  and maxiter (default = 100) controls the maximum number 
   //  of iteration of the Brent algorithm
   //  If the flag logx is set the grid search is done in log step size
   //  This is done automatically if the log scale is set in the current Pad
    //
   // NOTE: see also TF1::GetX
   
   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}

   if (!logx && gPad != 0) logx = gPad->GetLogx(); 

   ROOT::Math::BrentMinimizer1D bm;
   GInverseFunc g(this);
   ROOT::Math::WrappedFunction<GInverseFunc> wf1(g);
   bm.SetFunction( wf1, xmin, xmax );
   bm.SetNpx(fNpx);
   bm.SetLogScan(logx);
   bm.Minimize(maxiter, epsilon, epsilon );
   Double_t x;
   x = bm.XMinimum();

   return x;
}


//______________________________________________________________________________
Double_t TF1::GetMinimum(Double_t xmin, Double_t xmax, Double_t epsilon, Int_t maxiter, Bool_t logx) const
{
   // Returns the minimum value of the function on the (xmin, xmax) interval
   // Method:
   //  First, the grid search is used to bracket the maximum
   //  with the step size = (xmax-xmin)/fNpx. This way, the step size
   //  can be controlled via the SetNpx() function. If the function is
   //  unimodal or if its extrema are far apart, setting the fNpx to
   //  a small value speeds the algorithm up many times.
   //  Then, Brent's method is applied on the bracketed interval
   //  epsilon (default = 1.E-10) controls the relative accuracy (if |x| > 1 ) 
   //  and absolute (if |x| < 1)  and maxiter (default = 100) controls the maximum number 
   //  of iteration of the Brent algorithm
   //  If the flag logx is set the grid search is done in log step size
   //  This is done automatically if the log scale is set in the current Pad
   //
   // NOTE: see also TF1::GetMaximumX and TF1::GetX

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}

   if (!logx && gPad != 0) logx = gPad->GetLogx(); 

   ROOT::Math::BrentMinimizer1D bm;
   ROOT::Math::WrappedFunction<const TF1&> wf1(*this);
   bm.SetFunction( wf1, xmin, xmax );
   bm.SetNpx(fNpx);
   bm.SetLogScan(logx);
   bm.Minimize(maxiter, epsilon, epsilon );
   Double_t x;
   x = bm.FValMinimum();

   return x;
}


//______________________________________________________________________________
Double_t TF1::GetMinimumX(Double_t xmin, Double_t xmax, Double_t epsilon, Int_t maxiter, Bool_t logx) const
{
   // Returns the X value corresponding to the minimum value of the function
   // on the (xmin, xmax) interval
   // Method:
   //  First, the grid search is used to bracket the maximum
   //  with the step size = (xmax-xmin)/fNpx. This way, the step size
   //  can be controlled via the SetNpx() function. If the function is
   //  unimodal or if its extrema are far apart, setting the fNpx to
   //  a small value speeds the algorithm up many times.
   //  Then, Brent's method is applied on the bracketed interval
   //  epsilon (default = 1.E-10) controls the relative accuracy (if |x| > 1 ) 
   //  and absolute (if |x| < 1)  and maxiter (default = 100) controls the maximum number 
   //  of iteration of the Brent algorithm
   //  If the flag logx is set the grid search is done in log step size
   //  This is done automatically if the log scale is set in the current Pad
   //
   // NOTE: see also TF1::GetX

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}

   ROOT::Math::BrentMinimizer1D bm;
   ROOT::Math::WrappedFunction<const TF1&> wf1(*this);
   bm.SetFunction( wf1, xmin, xmax );
   bm.SetNpx(fNpx);
   bm.SetLogScan(logx);
   bm.Minimize(maxiter, epsilon, epsilon );
   Double_t x;
   x = bm.XMinimum();

   return x;
}


//______________________________________________________________________________
Double_t TF1::GetX(Double_t fy, Double_t xmin, Double_t xmax, Double_t epsilon, Int_t maxiter, Bool_t logx) const
{
   // Returns the X value corresponding to the function value fy for (xmin<x<xmax).
   // in other words it can find the roots of the function when fy=0 and successive calls
   // by changing the next call to [xmin+eps,xmax] where xmin is the previous root.
   // Method:
   //  First, the grid search is used to bracket the maximum
   //  with the step size = (xmax-xmin)/fNpx. This way, the step size
   //  can be controlled via the SetNpx() function. If the function is
   //  unimodal or if its extrema are far apart, setting the fNpx to
   //  a small value speeds the algorithm up many times.
   //  Then, Brent's method is applied on the bracketed interval
   //  epsilon (default = 1.E-10) controls the relative accuracy (if |x| > 1 ) 
   //  and absolute (if |x| < 1)  and maxiter (default = 100) controls the maximum number 
   //  of iteration of the Brent algorithm
   //  If the flag logx is set the grid search is done in log step size
   //  This is done automatically if the log scale is set in the current Pad
   //
   // NOTE: see also TF1::GetMaximumX, TF1::GetMinimumX

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}

   if (!logx && gPad != 0) logx = gPad->GetLogx(); 

   GFunc g(this, fy);
   ROOT::Math::WrappedFunction<GFunc> wf1(g);
   ROOT::Math::BrentRootFinder brf;
   brf.SetFunction(wf1,xmin,xmax);
   brf.SetNpx(fNpx);
   brf.SetLogScan(logx);
   brf.Solve(maxiter, epsilon, epsilon);
   return brf.Root();

}

//______________________________________________________________________________
Int_t TF1::GetNDF() const
{
   // Return the number of degrees of freedom in the fit
   // the fNDF parameter has been previously computed during a fit.
   // The number of degrees of freedom corresponds to the number of points
   // used in the fit minus the number of free parameters.

   if (fNDF == 0 && (fNpfits > fNpar) ) return fNpfits-fNpar;
   return fNDF;
}


//______________________________________________________________________________
Int_t TF1::GetNumberFreeParameters() const
{
   // Return the number of free parameters

   Int_t nfree = fNpar;
   Double_t al,bl;
   for (Int_t i=0;i<fNpar;i++) {
      ((TF1*)this)->GetParLimits(i,al,bl);
      if (al*bl != 0 && al >= bl) nfree--;
   }
   return nfree;
}


//______________________________________________________________________________
char *TF1::GetObjectInfo(Int_t px, Int_t /* py */) const
{
   // Redefines TObject::GetObjectInfo.
   // Displays the function info (x, function value)
   // corresponding to cursor position px,py

   static char info[64];
   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   snprintf(info,64,"(x=%g, f=%g)",x,((TF1*)this)->Eval(x));
   return info;
}


//______________________________________________________________________________
Double_t TF1::GetParError(Int_t ipar) const
{
   // Return value of parameter number ipar

   if (ipar < 0 || ipar > fNpar-1) return 0;
   return fParErrors[ipar];
}


//______________________________________________________________________________
void TF1::GetParLimits(Int_t ipar, Double_t &parmin, Double_t &parmax) const
{
   // Return limits for parameter ipar.

   parmin = 0;
   parmax = 0;
   if (ipar < 0 || ipar > fNpar-1) return;
   if (fParMin) parmin = fParMin[ipar];
   if (fParMax) parmax = fParMax[ipar];
}


//______________________________________________________________________________
Double_t TF1::GetProb() const
{
   // Return the fit probability

   if (fNDF <= 0) return 0;
   return TMath::Prob(fChisquare,fNDF);
}


//______________________________________________________________________________
Int_t TF1::GetQuantiles(Int_t nprobSum, Double_t *q, const Double_t *probSum)
{
   //  Compute Quantiles for density distribution of this function
   //     Quantile x_q of a probability distribution Function F is defined as
   //Begin_Latex
   //        F(x_{q}) = #int_{xmin}^{x_{q}} f dx = q with 0 <= q <= 1.
   //End_Latex
   //     For instance the median Begin_Latex x_{#frac{1}{2}} End_Latex of a distribution is defined as that value
   //     of the random variable for which the distribution function equals 0.5:
   //Begin_Latex
   //        F(x_{#frac{1}{2}}) = #prod(x < x_{#frac{1}{2}}) = #frac{1}{2}
   //End_Latex
   //  code from Eddy Offermann, Renaissance
   //
   // input parameters
   //   - this TF1 function
   //   - nprobSum maximum size of array q and size of array probSum
   //   - probSum array of positions where quantiles will be computed.
   //     It is assumed to contain at least nprobSum values.
   //  output
   //   - return value nq (<=nprobSum) with the number of quantiles computed
   //   - array q filled with nq quantiles
   //
   //  Getting quantiles from two histograms and storing results in a TGraph,
   //   a so-called QQ-plot
   //
   //     TGraph *gr = new TGraph(nprob);
   //     f1->GetQuantiles(nprob,gr->GetX());
   //     f2->GetQuantiles(nprob,gr->GetY());
   //     gr->Draw("alp");

   // LM: change to use fNpx 
   // should we change code to use a root finder ? 
   // It should be more precise and more efficient
   const Int_t npx     = TMath::Max(fNpx, 2*nprobSum);
   const Double_t xMin = GetXmin();
   const Double_t xMax = GetXmax();
   const Double_t dx   = (xMax-xMin)/npx;

   TArrayD integral(npx+1);
   TArrayD alpha(npx);
   TArrayD beta(npx);
   TArrayD gamma(npx);

   integral[0] = 0;
   Int_t intNegative = 0;
   Int_t i;
   for (i = 0; i < npx; i++) {
      const Double_t *params = 0;
      Double_t integ = Integral(Double_t(xMin+i*dx),Double_t(xMin+i*dx+dx),params);
      if (integ < 0) {intNegative++; integ = -integ;}
      integral[i+1] = integral[i] + integ;
   }

   if (intNegative > 0)
      Warning("GetQuantiles","function:%s has %d negative values: abs assumed",
      GetName(),intNegative);
   if (integral[npx] == 0) {
      Error("GetQuantiles","Integral of function is zero");
      return 0;
   }

   const Double_t total = integral[npx];
   for (i = 1; i <= npx; i++) integral[i] /= total;
   //the integral r for each bin is approximated by a parabola
   //  x = alpha + beta*r +gamma*r**2
   // compute the coefficients alpha, beta, gamma for each bin
   for (i = 0; i < npx; i++) {
      const Double_t x0 = xMin+dx*i;
      const Double_t r2 = integral[i+1]-integral[i];
      const Double_t r1 = Integral(x0,x0+0.5*dx)/total;
      gamma[i] = (2*r2-4*r1)/(dx*dx);
      beta[i]  = r2/dx-gamma[i]*dx;
      alpha[i] = x0;
      gamma[i] *= 2;
   }

   // Be careful because of finite precision in the integral; Use the fact that the integral
   // is monotone increasing
   for (i = 0; i < nprobSum; i++) {
      const Double_t r = probSum[i];
      Int_t bin  = TMath::Max(TMath::BinarySearch(npx+1,integral.GetArray(),r),(Long64_t)0);
      // LM use a tolerance 1.E-12 (integral precision)
      while (bin < npx-1 && TMath::AreEqualRel(integral[bin+1], r, 1E-12) ) {
         if (TMath::AreEqualRel(integral[bin+2], r, 1E-12) ) bin++;
         else break;
      }

      const Double_t rr = r-integral[bin];
      if (rr != 0.0) {
         Double_t xx = 0.0;
         const Double_t fac = -2.*gamma[bin]*rr/beta[bin]/beta[bin];
         if (fac != 0 && fac <= 1)
            xx = (-beta[bin]+TMath::Sqrt(beta[bin]*beta[bin]+2*gamma[bin]*rr))/gamma[bin];
         else if (beta[bin] != 0.)
            xx = rr/beta[bin];
         q[i] = alpha[bin]+xx;
      } else {
         q[i] = alpha[bin];
         if (integral[bin+1] == r) q[i] += dx;
      }
   }

   return nprobSum;
}


//______________________________________________________________________________
Double_t TF1::GetRandom()
{
   // Return a random number following this function shape
   //
   //   The distribution contained in the function fname (TF1) is integrated
   //   over the channel contents.
   //   It is normalized to 1.
   //   For each bin the integral is approximated by a parabola.
   //   The parabola coefficients are stored as non persistent data members
   //   Getting one random number implies:
   //     - Generating a random number between 0 and 1 (say r1)
   //     - Look in which bin in the normalized integral r1 corresponds to
   //     - Evaluate the parabolic curve in the selected bin to find
   //       the corresponding X value.
   //   if the ratio fXmax/fXmin > fNpx the integral is tabulated in log scale in x
   //   The parabolic approximation is very good as soon as the number
   //   of bins is greater than 50.

   //  Check if integral array must be build
   if (fIntegral == 0) {
      fIntegral = new Double_t[fNpx+1];
      fAlpha    = new Double_t[fNpx+1];
      fBeta     = new Double_t[fNpx];
      fGamma    = new Double_t[fNpx];
      fIntegral[0] = 0;
      fAlpha[fNpx] = 0;
      Double_t integ;
      Int_t intNegative = 0;
      Int_t i;
      Bool_t logbin = kFALSE;
      Double_t dx;
      Double_t xmin = fXmin;
      Double_t xmax = fXmax;
      if (xmin > 0 && xmax/xmin> fNpx) {
         logbin =  kTRUE;
         fAlpha[fNpx] = 1;
         xmin = TMath::Log10(fXmin);
         xmax = TMath::Log10(fXmax);
      }
      dx = (xmax-xmin)/fNpx;
         
      Double_t *xx = new Double_t[fNpx+1];
      for (i=0;i<fNpx;i++) {
            xx[i] = xmin +i*dx;
      }
      xx[fNpx] = xmax;
      for (i=0;i<fNpx;i++) {
         if (logbin) {
            integ = Integral(TMath::Power(10,xx[i]), TMath::Power(10,xx[i+1]));
         } else {
            integ = Integral(xx[i],xx[i+1]);
         }
         if (integ < 0) {intNegative++; integ = -integ;}
         fIntegral[i+1] = fIntegral[i] + integ;
      }
      if (intNegative > 0) {
         Warning("GetRandom","function:%s has %d negative values: abs assumed",GetName(),intNegative);
      }
      if (fIntegral[fNpx] == 0) {
         delete [] xx;
         Error("GetRandom","Integral of function is zero");
         return 0;
      }
      Double_t total = fIntegral[fNpx];
      for (i=1;i<=fNpx;i++) {  // normalize integral to 1
         fIntegral[i] /= total;
      }
      //the integral r for each bin is approximated by a parabola
      //  x = alpha + beta*r +gamma*r**2
      // compute the coefficients alpha, beta, gamma for each bin
      Double_t x0,r1,r2,r3;
      for (i=0;i<fNpx;i++) {
         x0 = xx[i];
         r2 = fIntegral[i+1] - fIntegral[i];
         if (logbin) r1 = Integral(TMath::Power(10,x0),TMath::Power(10,x0+0.5*dx))/total;
         else        r1 = Integral(x0,x0+0.5*dx)/total;
         r3 = 2*r2 - 4*r1;
         if (TMath::Abs(r3) > 1e-8) fGamma[i] = r3/(dx*dx);
         else           fGamma[i] = 0;
         fBeta[i]  = r2/dx - fGamma[i]*dx;
         fAlpha[i] = x0;
         fGamma[i] *= 2;
      }
      delete [] xx;
   }

   // return random number
   Double_t r  = gRandom->Rndm();
   Int_t bin  = TMath::BinarySearch(fNpx,fIntegral,r);
   Double_t rr = r - fIntegral[bin];

   Double_t yy;
   if(fGamma[bin] != 0)
      yy = (-fBeta[bin] + TMath::Sqrt(fBeta[bin]*fBeta[bin]+2*fGamma[bin]*rr))/fGamma[bin];
   else
      yy = rr/fBeta[bin];
   Double_t x = fAlpha[bin] + yy;
   if (fAlpha[fNpx] > 0) return TMath::Power(10,x);
   return x;
}


//______________________________________________________________________________
Double_t TF1::GetRandom(Double_t xmin, Double_t xmax)
{
   // Return a random number following this function shape in [xmin,xmax]
   //
   //   The distribution contained in the function fname (TF1) is integrated
   //   over the channel contents.
   //   It is normalized to 1.
   //   For each bin the integral is approximated by a parabola.
   //   The parabola coefficients are stored as non persistent data members
   //   Getting one random number implies:
   //     - Generating a random number between 0 and 1 (say r1)
   //     - Look in which bin in the normalized integral r1 corresponds to
   //     - Evaluate the parabolic curve in the selected bin to find
   //       the corresponding X value.
   //   The parabolic approximation is very good as soon as the number
   //   of bins is greater than 50.
   //
   //  IMPORTANT NOTE
   //  The integral of the function is computed at fNpx points. If the function
   //  has sharp peaks, you should increase the number of points (SetNpx)
   //  such that the peak is correctly tabulated at several points.

   //  Check if integral array must be build
   if (fIntegral == 0) {
      Double_t dx = (fXmax-fXmin)/fNpx;
      fIntegral = new Double_t[fNpx+1];
      fAlpha    = new Double_t[fNpx];
      fBeta     = new Double_t[fNpx];
      fGamma    = new Double_t[fNpx];
      fIntegral[0] = 0;
      Double_t integ;
      Int_t intNegative = 0;
      Int_t i;
      for (i=0;i<fNpx;i++) {
         integ = Integral(Double_t(fXmin+i*dx), Double_t(fXmin+i*dx+dx));
         if (integ < 0) {intNegative++; integ = -integ;}
         fIntegral[i+1] = fIntegral[i] + integ;
      }
      if (intNegative > 0) {
         Warning("GetRandom","function:%s has %d negative values: abs assumed",GetName(),intNegative);
      }
      if (fIntegral[fNpx] == 0) {
         Error("GetRandom","Integral of function is zero");
         return 0;
      }
      Double_t total = fIntegral[fNpx];
      for (i=1;i<=fNpx;i++) {  // normalize integral to 1
         fIntegral[i] /= total;
      }
      //the integral r for each bin is approximated by a parabola
      //  x = alpha + beta*r +gamma*r**2
      // compute the coefficients alpha, beta, gamma for each bin
      Double_t x0,r1,r2,r3;
      for (i=0;i<fNpx;i++) {
         x0 = fXmin+i*dx;
         r2 = fIntegral[i+1] - fIntegral[i];
         r1 = Integral(x0,x0+0.5*dx)/total;
         r3 = 2*r2 - 4*r1;
         if (TMath::Abs(r3) > 1e-8) fGamma[i] = r3/(dx*dx);
         else           fGamma[i] = 0;
         fBeta[i]  = r2/dx - fGamma[i]*dx;
         fAlpha[i] = x0;
         fGamma[i] *= 2;
      }
   }

   // return random number
   Double_t dx   = (fXmax-fXmin)/fNpx;
   Int_t nbinmin = (Int_t)((xmin-fXmin)/dx);
   Int_t nbinmax = (Int_t)((xmax-fXmin)/dx)+2;
   if(nbinmax>fNpx) nbinmax=fNpx;

   Double_t pmin=fIntegral[nbinmin];
   Double_t pmax=fIntegral[nbinmax];

   Double_t r,x,xx,rr;
   do {
      r  = gRandom->Uniform(pmin,pmax);

      Int_t bin  = TMath::BinarySearch(fNpx,fIntegral,r);
      rr = r - fIntegral[bin];

      if(fGamma[bin] != 0)
         xx = (-fBeta[bin] + TMath::Sqrt(fBeta[bin]*fBeta[bin]+2*fGamma[bin]*rr))/fGamma[bin];
      else
         xx = rr/fBeta[bin];
      x = fAlpha[bin] + xx;
   } while(x<xmin || x>xmax);
   return x;
}


//______________________________________________________________________________
void TF1::GetRange(Double_t &xmin, Double_t &xmax) const
{
   // Return range of a 1-D function.

   xmin = fXmin;
   xmax = fXmax;
}


//______________________________________________________________________________
void TF1::GetRange(Double_t &xmin, Double_t &ymin,  Double_t &xmax, Double_t &ymax) const
{
   // Return range of a 2-D function.

   xmin = fXmin;
   xmax = fXmax;
   ymin = 0;
   ymax = 0;
}


//______________________________________________________________________________
void TF1::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const
{
   // Return range of function.

   xmin = fXmin;
   xmax = fXmax;
   ymin = 0;
   ymax = 0;
   zmin = 0;
   zmax = 0;
}


//______________________________________________________________________________
Double_t TF1::GetSave(const Double_t *xx)
{
    // Get value corresponding to X in array of fSave values

   if (fNsave <= 0) return 0;
   if (fSave == 0) return 0;
   Double_t x    = Double_t(xx[0]);
   Double_t y,dx,xmin,xmax,xlow,xup,ylow,yup;
   if (fParent && fParent->InheritsFrom(TH1::Class())) {
      //if parent is a histogram the function had been savedat the center of the bins
      //we make a linear interpolation between the saved values
      xmin = fSave[fNsave-3];
      xmax = fSave[fNsave-2];
      if (fSave[fNsave-1] == xmax) {
         TH1 *h = (TH1*)fParent;
         TAxis *xaxis = h->GetXaxis();
         Int_t bin1  = xaxis->FindBin(xmin);
         Int_t binup = xaxis->FindBin(xmax);
         Int_t bin   = xaxis->FindBin(x);
         if (bin < binup) {
            xlow = xaxis->GetBinCenter(bin);
            xup  = xaxis->GetBinCenter(bin+1);
            ylow = fSave[bin-bin1];
            yup  = fSave[bin-bin1+1];
         } else {
            xlow = xaxis->GetBinCenter(bin-1);
            xup  = xaxis->GetBinCenter(bin);
            ylow = fSave[bin-bin1-1];
            yup  = fSave[bin-bin1];
         }
         dx = xup-xlow;
         y  = ((xup*ylow-xlow*yup) + x*(yup-ylow))/dx;
         return y;
      }
   }
   Int_t np = fNsave - 3;
   xmin = Double_t(fSave[np+1]);
   xmax = Double_t(fSave[np+2]);
   dx   = (xmax-xmin)/np;
   if (x < xmin || x > xmax) return 0;
   if (dx <= 0) return 0;

   Int_t bin     = Int_t((x-xmin)/dx);
   xlow = xmin + bin*dx;
   xup  = xlow + dx;
   ylow = fSave[bin];
   yup  = fSave[bin+1];
   y    = ((xup*ylow-xlow*yup) + x*(yup-ylow))/dx;
   return y;
}


//______________________________________________________________________________
TAxis *TF1::GetXaxis() const
{
   // Get x axis of the function.

   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetXaxis();
}


//______________________________________________________________________________
TAxis *TF1::GetYaxis() const
{
   // Get y axis of the function.

   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetYaxis();
}


//______________________________________________________________________________
TAxis *TF1::GetZaxis() const
{
   // Get z axis of the function. (In case this object is a TF2 or TF3)

   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetZaxis();
}



//______________________________________________________________________________
Double_t TF1::GradientPar(Int_t ipar, const Double_t *x, Double_t eps)
{
   // Compute the gradient (derivative) wrt a parameter ipar
   // Parameters:
   // ipar - index of parameter for which the derivative is computed
   // x - point, where the derivative is computed
   // eps - if the errors of parameters have been computed, the step used in
   // numerical differentiation is eps*parameter_error.
   // if the errors have not been computed, step=eps is used
   // default value of eps = 0.01
   // Method is the same as in Derivative() function
   //
   // If a paramter is fixed, the gradient on this parameter = 0

   if (fNpar == 0) return 0; 

   if(eps< 1e-10 || eps > 1) {
      Warning("Derivative","parameter esp=%g out of allowed range[1e-10,1], reset to 0.01",eps);
      eps = 0.01;
   }
   Double_t h;
   TF1 *func = (TF1*)this;
   //save original parameters
   Double_t par0 = fParams[ipar];


   func->InitArgs(x, fParams);

   Double_t al, bl;
   Double_t f1, f2, g1, g2, h2, d0, d2;

   ((TF1*)this)->GetParLimits(ipar,al,bl);
   if (al*bl != 0 && al >= bl) {
      //this parameter is fixed
      return 0;
   }

   // check if error has been computer (is not zero)
   if (func->GetParError(ipar)!=0)
      h = eps*func->GetParError(ipar);
   else
      h=eps;



   fParams[ipar] = par0 + h;     f1 = func->EvalPar(x,fParams);
   fParams[ipar] = par0 - h;     f2 = func->EvalPar(x,fParams);
   fParams[ipar] = par0 + h/2;   g1 = func->EvalPar(x,fParams);
   fParams[ipar] = par0 - h/2;   g2 = func->EvalPar(x,fParams);

   //compute the central differences
   h2    = 1/(2.*h);
   d0    = f1 - f2;
   d2    = 2*(g1 - g2);

   Double_t  grad = h2*(4*d2 - d0)/3.;

   // restore original value
   fParams[ipar] = par0;

   return grad;
}

//______________________________________________________________________________
void TF1::GradientPar(const Double_t *x, Double_t *grad, Double_t eps)
{
   // Compute the gradient wrt parameters
   // Parameters:
   // x - point, were the gradient is computed
   // grad - used to return the computed gradient, assumed to be of at least fNpar size
   // eps - if the errors of parameters have been computed, the step used in
   // numerical differentiation is eps*parameter_error.
   // if the errors have not been computed, step=eps is used
   // default value of eps = 0.01
   // Method is the same as in Derivative() function
   //
   // If a paramter is fixed, the gradient on this parameter = 0

   if(eps< 1e-10 || eps > 1) {
      Warning("Derivative","parameter esp=%g out of allowed range[1e-10,1], reset to 0.01",eps);
      eps = 0.01;
   }

   for (Int_t ipar=0; ipar<fNpar; ipar++){
      grad[ipar] = GradientPar(ipar,x,eps);
   }
}

//______________________________________________________________________________
void TF1::InitArgs(const Double_t *x, const Double_t *params)
{
   // Initialize parameters addresses.

   if (fMethodCall) {
      Long_t args[2];
      args[0] = (Long_t)x;
      if (params) args[1] = (Long_t)params;
      else        args[1] = (Long_t)fParams;
      fMethodCall->SetParamPtrs(args);
   }
}


//______________________________________________________________________________
void TF1::InitStandardFunctions()
{
   // Create the basic function objects

   TF1 *f1;
   if (!gROOT->GetListOfFunctions()->FindObject("gaus")) {
      f1 = new TF1("gaus","gaus",-1,1);       f1->SetParameters(1,0,1);
      f1 = new TF1("gausn","gausn",-1,1);     f1->SetParameters(1,0,1);
      f1 = new TF1("landau","landau",-1,1);   f1->SetParameters(1,0,1);
      f1 = new TF1("landaun","landaun",-1,1); f1->SetParameters(1,0,1);
      f1 = new TF1("expo","expo",-1,1);       f1->SetParameters(1,1);
      for (Int_t i=0;i<10;i++) {
         f1 = new TF1(Form("pol%d",i),Form("pol%d",i),-1,1);
         f1->SetParameters(1,1,1,1,1,1,1,1,1,1);
      }
   }
}


//______________________________________________________________________________
Double_t TF1::Integral(Double_t a, Double_t b, const Double_t *params, Double_t epsilon)
{
   // Return Integral of function between a and b.
   //
   //   based on original CERNLIB routine DGAUSS by Sigfried Kolbig
   //   converted to C++ by Rene Brun
   //
   // This function computes, to an attempted specified accuracy, the value
   // of the integral.
   //Begin_Latex
   //   I = #int^{B}_{A} f(x)dx
   //End_Latex
   // Usage:
   //   In any arithmetic expression, this function has the approximate value
   //   of the integral I.
   //   - A, B: End-points of integration interval. Note that B may be less
   //           than A.
   //   - params: Array of function parameters. If 0, use current parameters.
   //   - epsilon: Accuracy parameter (see Accuracy).
   //
   //Method:
   //   For any interval [a,b] we define g8(a,b) and g16(a,b) to be the 8-point
   //   and 16-point Gaussian quadrature approximations to
   //Begin_Latex
   //   I = #int^{b}_{a} f(x)dx
   //End_Latex
   //   and define
   //Begin_Latex
   //   r(a,b) = #frac{#||{g_{16}(a,b)-g_{8}(a,b)}}{1+#||{g_{16}(a,b)}}
   //End_Latex
   //   Then,
   //Begin_Latex
   //   G = #sum_{i=1}^{k}g_{16}(x_{i-1},x_{i})
   //End_Latex
   //   where, starting with x0 = A and finishing with xk = B,
   //   the subdivision points xi(i=1,2,...) are given by
   //Begin_Latex
   //   x_{i} = x_{i-1} + #lambda(B-x_{i-1})
   //End_Latex
   //   Begin_Latex #lambdaEnd_Latex is equal to the first member of the
   //   sequence 1,1/2,1/4,... for which r(xi-1, xi) < EPS.
   //   If, at any stage in the process of subdivision, the ratio
   //Begin_Latex
   //   q = #||{#frac{x_{i}-x_{i-1}}{B-A}}
   //End_Latex
   //   is so small that 1+0.005q is indistinguishable from 1 to
   //   machine accuracy, an error exit occurs with the function value
   //   set equal to zero.
   //
   // Accuracy:
   //   Unless there is severe cancellation of positive and negative values of
   //   f(x) over the interval [A,B], the argument EPS may be considered as
   //   specifying a bound on the <I>relative</I> error of I in the case
   //   |I|&gt;1, and a bound on the absolute error in the case |I|&lt;1. More
   //   precisely, if k is the number of sub-intervals contributing to the
   //   approximation (see Method), and if
   //Begin_Latex
   //   I_{abs} = #int^{B}_{A} #||{f(x)}dx
   //End_Latex
   //   then the relation
   //Begin_Latex
   //   #frac{#||{G-I}}{I_{abs}+k} < EPS
   //End_Latex
   //   will nearly always be true, provided the routine terminates without
   //   printing an error message. For functions f having no singularities in
   //   the closed interval [A,B] the accuracy will usually be much higher than
   //   this.
   //
   // Error handling:
   //   The requested accuracy cannot be obtained (see Method).
   //   The function value is set equal to zero.
   //
   // Note 1:
   //   Values of the function f(x) at the interval end-points A and B are not
   //   required. The subprogram may therefore be used when these values are
   //   undefined.
   //
   // Note 2:
   //   Instead of TF1::Integral, you may want to use the combination of
   //   TF1::CalcGaussLegendreSamplingPoints and TF1::IntegralFast.
   //   See an example with the following script:
   //
   //   void gint() {
   //      TF1 *g = new TF1("g","gaus",-5,5);
   //      g->SetParameters(1,0,1);
   //      //default gaus integration method uses 6 points
   //      //not suitable to integrate on a large domain
   //      double r1 = g->Integral(0,5);
   //      double r2 = g->Integral(0,1000);
   //
   //      //try with user directives computing more points
   //      Int_t np = 1000;
   //      double *x=new double[np];
   //      double *w=new double[np];
   //      g->CalcGaussLegendreSamplingPoints(np,x,w,1e-15);
   //      double r3 = g->IntegralFast(np,x,w,0,5);
   //      double r4 = g->IntegralFast(np,x,w,0,1000);
   //      double r5 = g->IntegralFast(np,x,w,0,10000);
   //      double r6 = g->IntegralFast(np,x,w,0,100000);
   //      printf("g->Integral(0,5)               = %g\n",r1);
   //      printf("g->Integral(0,1000)            = %g\n",r2);
   //      printf("g->IntegralFast(n,x,w,0,5)     = %g\n",r3);
   //      printf("g->IntegralFast(n,x,w,0,1000)  = %g\n",r4);
   //      printf("g->IntegralFast(n,x,w,0,10000) = %g\n",r5);
   //      printf("g->IntegralFast(n,x,w,0,100000)= %g\n",r6);
   //      delete [] x;
   //      delete [] w;
   //   }
   //
   //   This example produces the following results:
   //
   //      g->Integral(0,5)               = 1.25331
   //      g->Integral(0,1000)            = 1.25319
   //      g->IntegralFast(n,x,w,0,5)     = 1.25331
   //      g->IntegralFast(n,x,w,0,1000)  = 1.25331
   //      g->IntegralFast(n,x,w,0,10000) = 1.25331
   //      g->IntegralFast(n,x,w,0,100000)= 1.253


   TF1_EvalWrapper wf1( this, params, fgAbsValue ); 

   ROOT::Math::GaussIntegrator giod;
   //ROOT::Math::Integrator giod;
   giod.SetFunction(wf1);
   giod.SetRelTolerance(epsilon);
   //giod.SetAbsTolerance(epsilon);

   return giod.Integral(a, b);
}


//______________________________________________________________________________
Double_t TF1::Integral(Double_t, Double_t, Double_t, Double_t, Double_t)
{
   // Return Integral of a 2d function in range [ax,bx],[ay,by]

   Error("Integral","Must be called with a TF2 only");
   return 0;
}


//______________________________________________________________________________
Double_t TF1::Integral(Double_t, Double_t, Double_t, Double_t, Double_t, Double_t, Double_t)
{
   // Return Integral of a 3d function in range [ax,bx],[ay,by],[az,bz]

   Error("Integral","Must be called with a TF3 only");
   return 0;
}

//______________________________________________________________________________
Double_t TF1::IntegralError(Double_t a, Double_t b, const Double_t * params, const Double_t * covmat, Double_t epsilon )
{
   // Return Error on Integral of a parameteric function between a and b 
   // due to the parameter uncertainties.
   // A pointer to a vector of parameter values and to the elements of the covariance matrix (covmat)
   // can be optionally passed.  By default (i.e. when a zero pointer is passed) the current stored 
   // parameter values are used to estimate the integral error together with the covariance matrix
   // from the last fit (retrieved from the global fitter instance) 
   //
   // IMPORTANT NOTE1: When no covariance matrix is passed and in the meantime a fit is done 
   // using another function, the routine will signal an error and it will return zero only 
   // when the number of fit parameter is different than the values stored in TF1 (TF1::GetNpar() ). 
   // In the case that npar is the same, an incorrect result is returned. 
   //
   // IMPORTANT NOTE2: The user must pass a pointer to the elements of the full covariance matrix 
   // dimensioned with the right size (npar*npar), where npar is the total number of parameters (TF1::GetNpar()), 
   // including also the fixed parameters. When there are fixed parameters, the pointer returned from 
   // TVirtualFitter::GetCovarianceMatrix() cannot be used. 
   // One should use the TFitResult class, as shown in the example below.   
   // 
   // To get the matrix and values from an old fit do for example:  
   // TFitResultPtr r = histo->Fit(func, "S");
   // ..... after performing other fits on the same function do 
   // func->IntegralError(x1,x2,r->GetParams(), r->GetCovarianceMatrix()->GetMatrixArray() );

   Double_t x1[1]; 
   Double_t x2[1]; 
   x1[0] = a, x2[0] = b;
   return ROOT::TF1Helper::IntegralError(this,1,x1,x2,params,covmat,epsilon);
}

//______________________________________________________________________________
Double_t TF1::IntegralError(Int_t n, const Double_t * a, const Double_t * b, const Double_t * params, const  Double_t * covmat, Double_t epsilon )
{
   // Return Error on Integral of a parameteric function with dimension larger tan one 
   // between a[] and b[]  due to the parameters uncertainties.
   // For a TF1 with dimension larger than 1 (for example a TF2 or TF3) 
   // TF1::IntegralMultiple is used for the integral calculation
   //
   // A pointer to a vector of parameter values and to the elements of the covariance matrix (covmat) can be optionally passed.
   // By default (i.e. when a zero pointer is passed) the current stored parameter values are used to estimate the integral error 
   // together with the covariance matrix from the last fit (retrieved from the global fitter instance).
   //
   // IMPORTANT NOTE1: When no covariance matrix is passed and in the meantime a fit is done 
   // using another function, the routine will signal an error and it will return zero only 
   // when the number of fit parameter is different than the values stored in TF1 (TF1::GetNpar() ). 
   // In the case that npar is the same, an incorrect result is returned. 
   //
   // IMPORTANT NOTE2: The user must pass a pointer to the elements of the full covariance matrix 
   // dimensioned with the right size (npar*npar), where npar is the total number of parameters (TF1::GetNpar()), 
   // including also the fixed parameters. When there are fixed parameters, the pointer returned from 
   // TVirtualFitter::GetCovarianceMatrix() cannot be used. 
   // One should use the TFitResult class, as shown in the example below.   
   // 
   // To get the matrix and values from an old fit do for example:  
   // TFitResultPtr r = histo->Fit(func, "S");
   // ..... after performing other fits on the same function do 
   // func->IntegralError(x1,x2,r->GetParams(), r->GetCovarianceMatrix()->GetMatrixArray() );

   return ROOT::TF1Helper::IntegralError(this,n,a,b,params,covmat,epsilon);
}

#ifdef INTHEFUTURE
//______________________________________________________________________________
Double_t TF1::IntegralFast(const TGraph *g, Double_t a, Double_t b, Double_t *params)
{
   // Gauss-Legendre integral, see CalcGaussLegendreSamplingPoints

   if (!g) return 0;
   return IntegralFast(g->GetN(), g->GetX(), g->GetY(), a, b, params);
}
#endif


//______________________________________________________________________________
Double_t TF1::IntegralFast(Int_t num, Double_t * /* x */, Double_t * /* w */, Double_t a, Double_t b, Double_t *params, Double_t epsilon)
{
   // Gauss-Legendre integral, see CalcGaussLegendreSamplingPoints

   // Now x and w are not used!

   ROOT::Math::WrappedTF1 wf1(*this);
   if ( params )
      wf1.SetParameters( params );
   ROOT::Math::GaussLegendreIntegrator gli(num,epsilon);
   gli.SetFunction( wf1 );
   return gli.Integral(a, b);

}


//______________________________________________________________________________
Double_t TF1::IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Double_t eps, Double_t &relerr)
{
   //  See more general prototype below.
   //  This interface kept for back compatibility

   Int_t nfnevl,ifail;
   Int_t minpts = 2+2*n*(n+1)+1; //ie 7 for n=1
   Int_t maxpts = 1000;
   Double_t result = IntegralMultiple(n,a,b,minpts, maxpts,eps,relerr,nfnevl,ifail);
   if (ifail > 0) {
      Warning("IntegralMultiple","failed code=%d, ",ifail);
   }
   return result;
}


//______________________________________________________________________________
Double_t TF1::IntegralMultiple(Int_t n, const Double_t *a, const Double_t *b, Int_t minpts, Int_t maxpts, Double_t eps, Double_t &relerr,Int_t &nfnevl, Int_t &ifail)
{
   //  Adaptive Quadrature for Multiple Integrals over N-Dimensional
   //  Rectangular Regions
   //
   //Begin_Latex
   // I_{n} = #int_{a_{n}}^{b_{n}} #int_{a_{n-1}}^{b_{n-1}} ... #int_{a_{1}}^{b_{1}} f(x_{1}, x_{2},...,x_{n}) dx_{1}dx_{2}...dx_{n}
   //End_Latex
   //
   // Author(s): A.C. Genz, A.A. Malik
   // converted/adapted by R.Brun to C++ from Fortran CERNLIB routine RADMUL (D120)
   // The new code features many changes compared to the Fortran version.
   // Note that this function is currently called only by TF2::Integral (n=2)
   // and TF3::Integral (n=3).
   //
   // This function computes, to an attempted specified accuracy, the value of
   // the integral over an n-dimensional rectangular region.
   //
   // Input parameters:
   //
   //    n     : Number of dimensions [2,15]
   //    a,b   : One-dimensional arrays of length >= N . On entry A[i],  and  B[i],
   //            contain the lower and upper limits of integration, respectively.
   //    minpts: Minimum number of function evaluations requested. Must not exceed maxpts.
   //            if minpts < 1 minpts is set to 2^n +2*n*(n+1) +1
   //    maxpts: Maximum number of function evaluations to be allowed.
   //            maxpts >= 2^n +2*n*(n+1) +1
   //            if maxpts<minpts, maxpts is set to 10*minpts
   //    eps   : Specified relative accuracy.
   //
   // Output parameters:
   //
   //    relerr : Contains, on exit, an estimation of the relative accuracy of the result.
   //    nfnevl : number of function evaluations performed.
   //    ifail  :
   //        0 Normal exit.  . At least minpts and at most maxpts calls to the function were performed.
   //        1 maxpts is too small for the specified accuracy eps.
   //          The result and relerr contain the values obtainable for the
   //          specified value of maxpts.
   //        3 n<2 or n>15
   //
   // Method:
   //
   //    An integration rule of degree seven is used together with a certain
   //    strategy of subdivision.
   //    For a more detailed description of the method see References.
   //
   // Notes:
   //
   //   1.Multi-dimensional integration is time-consuming. For each rectangular
   //     subregion, the routine requires function evaluations.
   //     Careful programming of the integrand might result in substantial saving
   //     of time.
   //   2.Numerical integration usually works best for smooth functions.
   //     Some analysis or suitable transformations of the integral prior to
   //     numerical work may contribute to numerical efficiency.
   //
   // References:
   //
   //   1.A.C. Genz and A.A. Malik, Remarks on algorithm 006:
   //     An adaptive algorithm for numerical integration over
   //     an N-dimensional rectangular region, J. Comput. Appl. Math. 6 (1980) 295-302.
   //   2.A. van Doren and L. de Ridder, An adaptive algorithm for numerical
   //     integration over an n-dimensional cube, J.Comput. Appl. Math. 2 (1976) 207-217.

   ROOT::Math::WrappedMultiFunction<TF1&> wf1(*this, n);

   ROOT::Math::AdaptiveIntegratorMultiDim aimd(wf1, eps, eps, maxpts);
   aimd.SetMinPts(minpts);
   double result = aimd.Integral(a,b);
   relerr = aimd.RelError();
   nfnevl = aimd.NEval();
   ifail = 0;

   return result;
}


//______________________________________________________________________________
Bool_t TF1::IsInside(const Double_t *x) const
{
   // Return kTRUE if the point is inside the function range

   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   return kTRUE;
}


//______________________________________________________________________________
void TF1::Paint(Option_t *option)
{
   // Paint this function with its current attributes.

   Int_t i;
   Double_t xv[1];

   fgCurrent = this;

   TString opt = option;
   opt.ToLower();
   Bool_t optSAME = kFALSE;
   if (opt.Contains("same")) optSAME = kTRUE;

   Double_t xmin=fXmin, xmax=fXmax, pmin=fXmin, pmax=fXmax;
   if (gPad) {
      pmin = gPad->PadtoX(gPad->GetUxmin());
      pmax = gPad->PadtoX(gPad->GetUxmax());
   }
   if (optSAME) {
      if (xmax < pmin) return;  // Completely outside.
      if (xmin > pmax) return;
      if (xmin < pmin) xmin = pmin;
      if (xmax > pmax) xmax = pmax;
   }

   //  Create a temporary histogram and fill each channel with the function value
   //  Preserve axis titles
   TString xtitle = "";
   TString ytitle = "";
   char *semicol = (char*)strstr(GetTitle(),";");
   if (semicol) {
      Int_t nxt = strlen(semicol);
      char *ctemp = new char[nxt];
      strlcpy(ctemp,semicol+1,nxt);
      semicol = (char*)strstr(ctemp,";");
      if (semicol) {
         *semicol = 0;
         ytitle = semicol+1;
      }
      xtitle = ctemp;
      delete [] ctemp;
   }
   if (fHistogram) {
      xtitle = fHistogram->GetXaxis()->GetTitle();
      ytitle = fHistogram->GetYaxis()->GetTitle();
      if (!gPad->GetLogx()  &&  fHistogram->TestBit(TH1::kLogX)) { delete fHistogram; fHistogram = 0;}
      if ( gPad->GetLogx()  && !fHistogram->TestBit(TH1::kLogX)) { delete fHistogram; fHistogram = 0;}
   }

   if (fHistogram) {
      fHistogram->GetXaxis()->SetLimits(xmin,xmax);
   } else {
      // If logx, we must bin in logx and not in x
      // otherwise in case of several decades, one gets wrong results.
      if (xmin > 0 && gPad && gPad->GetLogx()) {
         Double_t *xbins    = new Double_t[fNpx+1];
         Double_t xlogmin = TMath::Log10(xmin);
         Double_t xlogmax = TMath::Log10(xmax);
         Double_t dlogx   = (xlogmax-xlogmin)/((Double_t)fNpx);
         for (i=0;i<=fNpx;i++) {
            xbins[i] = gPad->PadtoX(xlogmin+ i*dlogx);
         }
         fHistogram = new TH1D("Func",GetTitle(),fNpx,xbins);
         fHistogram->SetBit(TH1::kLogX);
         delete [] xbins;
      } else {
         fHistogram = new TH1D("Func",GetTitle(),fNpx,xmin,xmax);
      }
      if (!fHistogram) return;
      if (fMinimum != -1111) fHistogram->SetMinimum(fMinimum);
      if (fMaximum != -1111) fHistogram->SetMaximum(fMaximum);
      fHistogram->SetDirectory(0);
   }
   // Restore axis titles.
   fHistogram->GetXaxis()->SetTitle(xtitle.Data());
   fHistogram->GetYaxis()->SetTitle(ytitle.Data());

   InitArgs(xv,fParams);
   for (i=1;i<=fNpx;i++) {
      xv[0] = fHistogram->GetBinCenter(i);
      fHistogram->SetBinContent(i,EvalPar(xv,fParams));
   }

   // Copy Function attributes to histogram attributes.
   Double_t minimum   = fHistogram->GetMinimumStored();
   Double_t maximum   = fHistogram->GetMaximumStored();
   if (minimum <= 0 && gPad && gPad->GetLogy()) minimum = -1111; // This can happen when switching from lin to log scale.
   if (gPad && gPad->GetUymin() < fHistogram->GetMinimum() &&
       !fHistogram->TestBit(TH1::kIsZoomed)) minimum = -1111; // This can happen after unzooming a fit.
   if (minimum == -1111) { // This can happen after unzooming.
      if (fHistogram->TestBit(TH1::kIsZoomed)) {
         minimum = fHistogram->GetYaxis()->GetXmin();
      } else {
         minimum = fMinimum;
         // Optimize the computation of the scale in Y in case the min/max of the 
         // function oscillate around a constant value
         if (minimum == -1111) {
            Double_t hmin;
            if (optSAME) hmin = gPad->GetUymin();
            else         hmin = fHistogram->GetMinimum();
            if (hmin > 0) {
               Double_t hmax;
               Double_t hminpos = hmin;
               if (optSAME) hmax = gPad->GetUymax();
               else         hmax = fHistogram->GetMaximum();
               hmin -= 0.05*(hmax-hmin);
               if (hmin < 0) hmin = 0;
               if (hmin <= 0 && gPad && gPad->GetLogy()) hmin = hminpos;
               minimum = hmin;
            }
         }
      }
      fHistogram->SetMinimum(minimum);
   }
   if (maximum == -1111) {
      if (fHistogram->TestBit(TH1::kIsZoomed)) {
         maximum = fHistogram->GetYaxis()->GetXmax();
      } else {
         maximum = fMaximum;
      }
      fHistogram->SetMaximum(maximum);
   }
   fHistogram->SetBit(TH1::kNoStats);
   fHistogram->SetLineColor(GetLineColor());
   fHistogram->SetLineStyle(GetLineStyle());
   fHistogram->SetLineWidth(GetLineWidth());
   fHistogram->SetFillColor(GetFillColor());
   fHistogram->SetFillStyle(GetFillStyle());
   fHistogram->SetMarkerColor(GetMarkerColor());
   fHistogram->SetMarkerStyle(GetMarkerStyle());
   fHistogram->SetMarkerSize(GetMarkerSize());

   // Draw the histogram.
   if (!gPad) return;
   if (opt.Length() == 0) fHistogram->Paint("lf");
   else if (optSAME)      fHistogram->Paint("lfsame");
   else                   fHistogram->Paint(option);
}


//______________________________________________________________________________
void TF1::Print(Option_t *option) const
{
   // Dump this function with its attributes.

   TFormula::Print(option);
   if (fHistogram) fHistogram->Print(option);
}


//______________________________________________________________________________
void TF1::ReleaseParameter(Int_t ipar)
{
   // Release parameter number ipar If used in a fit, the parameter
   // can vary freely. The parameter limits are reset to 0,0.

   if (ipar < 0 || ipar > fNpar-1) return;
   SetParLimits(ipar,0,0);
}


//______________________________________________________________________________
void TF1::Save(Double_t xmin, Double_t xmax, Double_t, Double_t, Double_t, Double_t)
{
   // Save values of function in array fSave

   if (fSave != 0) {delete [] fSave; fSave = 0;}
   if (fParent && fParent->InheritsFrom(TH1::Class())) {
      //if parent is a histogram save the function at the center of the bins
      if ((xmin >0 && xmax > 0) && TMath::Abs(TMath::Log10(xmax/xmin) > TMath::Log10(fNpx))) {
         TH1 *h = (TH1*)fParent;
         Int_t bin1 = h->GetXaxis()->FindBin(xmin);
         Int_t bin2 = h->GetXaxis()->FindBin(xmax);
         fNsave = bin2-bin1+4;
         fSave  = new Double_t[fNsave];
         Double_t xv[1];
         InitArgs(xv,fParams);
         for (Int_t i=bin1;i<=bin2;i++) {
            xv[0]    = h->GetXaxis()->GetBinCenter(i);
            fSave[i-bin1] = EvalPar(xv,fParams);
         }
         fSave[fNsave-3] = xmin;
         fSave[fNsave-2] = xmax;
         fSave[fNsave-1] = xmax;
         return;
      }
   }
   fNsave = fNpx+3;
   if (fNsave <= 3) {fNsave=0; return;}
   fSave  = new Double_t[fNsave];
   Double_t dx = (xmax-xmin)/fNpx;
   if (dx <= 0) {
      dx = (fXmax-fXmin)/fNpx;
      fNsave--;
      xmin = fXmin +0.5*dx;
      xmax = fXmax -0.5*dx;
   }
   Double_t xv[1];
   InitArgs(xv,fParams);
   for (Int_t i=0;i<=fNpx;i++) {
      xv[0]    = xmin + dx*i;
      fSave[i] = EvalPar(xv,fParams);
   }
   fSave[fNpx+1] = xmin;
   fSave[fNpx+2] = xmax;
}


//______________________________________________________________________________
void TF1::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out

   Int_t i;
   char quote = '"';
   out<<"   "<<endl;
   //if (!fMethodCall) {
   if (!fType) {
      out<<"   TF1 *"<<GetName()<<" = new TF1("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote<<","<<fXmin<<","<<fXmax<<");"<<endl;
      if (fNpx != 100) {
         out<<"   "<<GetName()<<"->SetNpx("<<fNpx<<");"<<endl;
      }
   } else {
      out<<"   TF1 *"<<GetName()<<" = new TF1("<<quote<<"*"<<GetName()<<quote<<","<<fXmin<<","<<fXmax<<","<<GetNpar()<<");"<<endl;
      out<<"    //The original function : "<<GetTitle()<<" had originally been created by:" <<endl;
      out<<"    //TF1 *"<<GetName()<<" = new TF1("<<quote<<GetName()<<quote<<","<<GetTitle()<<","<<fXmin<<","<<fXmax<<","<<GetNpar()<<");"<<endl;
      out<<"   "<<GetName()<<"->SetRange("<<fXmin<<","<<fXmax<<");"<<endl;
      out<<"   "<<GetName()<<"->SetName("<<quote<<GetName()<<quote<<");"<<endl;
      out<<"   "<<GetName()<<"->SetTitle("<<quote<<GetTitle()<<quote<<");"<<endl;
      if (fNpx != 100) {
         out<<"   "<<GetName()<<"->SetNpx("<<fNpx<<");"<<endl;
      }
      Double_t dx = (fXmax-fXmin)/fNpx;
      Double_t xv[1];
      InitArgs(xv,fParams);
      for (i=0;i<=fNpx;i++) {
         xv[0]    = fXmin + dx*i;
         Double_t save = EvalPar(xv,fParams);
         out<<"   "<<GetName()<<"->SetSavedPoint("<<i<<","<<save<<");"<<endl;
      }
      out<<"   "<<GetName()<<"->SetSavedPoint("<<fNpx+1<<","<<fXmin<<");"<<endl;
      out<<"   "<<GetName()<<"->SetSavedPoint("<<fNpx+2<<","<<fXmax<<");"<<endl;
   }

   if (TestBit(kNotDraw)) {
      out<<"   "<<GetName()<<"->SetBit(TF1::kNotDraw);"<<endl;
   }
   if (GetFillColor() != 0) {
      if (GetFillColor() > 228) {
         TColor::SaveColor(out, GetFillColor());
         out<<"   "<<GetName()<<"->SetFillColor(ci);" << endl;
      } else
         out<<"   "<<GetName()<<"->SetFillColor("<<GetFillColor()<<");"<<endl;
   }
   if (GetFillStyle() != 1001) {
      out<<"   "<<GetName()<<"->SetFillStyle("<<GetFillStyle()<<");"<<endl;
   }
   if (GetMarkerColor() != 1) {
      if (GetMarkerColor() > 228) {
         TColor::SaveColor(out, GetMarkerColor());
         out<<"   "<<GetName()<<"->SetMarkerColor(ci);" << endl;
      } else
         out<<"   "<<GetName()<<"->SetMarkerColor("<<GetMarkerColor()<<");"<<endl;
   }
   if (GetMarkerStyle() != 1) {
      out<<"   "<<GetName()<<"->SetMarkerStyle("<<GetMarkerStyle()<<");"<<endl;
   }
   if (GetMarkerSize() != 1) {
      out<<"   "<<GetName()<<"->SetMarkerSize("<<GetMarkerSize()<<");"<<endl;
   }
   if (GetLineColor() != 1) {
      if (GetLineColor() > 228) {
         TColor::SaveColor(out, GetLineColor());
         out<<"   "<<GetName()<<"->SetLineColor(ci);" << endl;
      } else
         out<<"   "<<GetName()<<"->SetLineColor("<<GetLineColor()<<");"<<endl;
   }
   if (GetLineWidth() != 4) {
      out<<"   "<<GetName()<<"->SetLineWidth("<<GetLineWidth()<<");"<<endl;
   }
   if (GetLineStyle() != 1) {
      out<<"   "<<GetName()<<"->SetLineStyle("<<GetLineStyle()<<");"<<endl;
   }
   if (GetChisquare() != 0) {
      out<<"   "<<GetName()<<"->SetChisquare("<<GetChisquare()<<");"<<endl;
      out<<"   "<<GetName()<<"->SetNDF("<<GetNDF()<<");"<<endl;
   }

   GetXaxis()->SaveAttributes(out,GetName(),"->GetXaxis()");
   GetYaxis()->SaveAttributes(out,GetName(),"->GetYaxis()");

   Double_t parmin, parmax;
   for (i=0;i<fNpar;i++) {
      out<<"   "<<GetName()<<"->SetParameter("<<i<<","<<GetParameter(i)<<");"<<endl;
      out<<"   "<<GetName()<<"->SetParError("<<i<<","<<GetParError(i)<<");"<<endl;
      GetParLimits(i,parmin,parmax);
      out<<"   "<<GetName()<<"->SetParLimits("<<i<<","<<parmin<<","<<parmax<<");"<<endl;
   }
   if (!strstr(option,"nodraw")) {
      out<<"   "<<GetName()<<"->Draw("
         <<quote<<option<<quote<<");"<<endl;
   }
}


//______________________________________________________________________________
void TF1::SetCurrent(TF1 *f1)
{
   // Static function setting the current function.
   // the current function may be accessed in static C-like functions
   // when fitting or painting a function.

   fgCurrent = f1;
}

//______________________________________________________________________________
void TF1::SetFitResult(const ROOT::Fit::FitResult & result, const Int_t* indpar )
{
   // Set the result from the fit  
   // parameter values, errors, chi2, etc...
   // Optionally a pointer to a vector (with size fNpar) of the parameter indices in the FitResult can be passed
   // This is useful in the case of a combined fit with different functions, and the FitResult contains the global result 
   // By default it is assume that indpar = {0,1,2,....,fNpar-1}. 

   if (result.IsEmpty()) { 
      Warning("SetFitResult","Empty Fit result - nathing is set in TF1");
      return;      
   }
   if (indpar == 0 && fNpar != (int) result.NPar() ) { 
      Error("SetFitResult","Invalid Fit result passed - number of parameter is %d , different than TF1::GetNpar() = %d",fNpar,result.NPar());
      return;
   }
   if (result.Chi2() > 0) 
      SetChisquare(result.Chi2() );
   else 
      SetChisquare(result.MinFcnValue() );

   SetNDF(result.Ndf() );
   SetNumberFitPoints(result.Ndf() + result.NFreeParameters() );


   for (Int_t i = 0; i < fNpar; ++i) { 
      Int_t ipar = (indpar != 0) ? indpar[i] : i;  
      if (ipar < 0) continue;
      fParams[i] = result.Parameter(ipar);
      // in case errors are not present do not set them
      if (ipar < (int) result.Errors().size() )
         fParErrors[i] = result.Error(ipar);
   }
   //invalidate cached integral since parameters have changed
   Update();   
         
}


//______________________________________________________________________________
void TF1::SetMaximum(Double_t maximum)
{
   // Set the maximum value along Y for this function
   // In case the function is already drawn, set also the maximum in the
   // helper histogram

   fMaximum = maximum;
   if (fHistogram) fHistogram->SetMaximum(maximum);
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TF1::SetMinimum(Double_t minimum)
{
   // Set the minimum value along Y for this function
   // In case the function is already drawn, set also the minimum in the
   // helper histogram

   fMinimum = minimum;
   if (fHistogram) fHistogram->SetMinimum(minimum);
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TF1::SetNDF(Int_t ndf)
{
   // Set the number of degrees of freedom
   // ndf should be the number of points used in a fit - the number of free parameters

   fNDF = ndf;
}


//______________________________________________________________________________
void TF1::SetNpx(Int_t npx)
{
   // Set the number of points used to draw the function
   //
   // The default number of points along x is 100 for 1-d functions and 30 for 2-d/3-d functions
   // You can increase this value to get a better resolution when drawing
   // pictures with sharp peaks or to get a better result when using TF1::GetRandom
   // the minimum number of points is 4, the maximum is 10000000 for 1-d and 10000 for 2-d/3-d functions

   const Int_t minPx = 4;
   Int_t maxPx = 10000000;
   if (GetNdim() > 1) maxPx = 10000;
   if (npx >= minPx && npx <= maxPx) {
      fNpx = npx;
   } 
   else { 
      if(npx < minPx) fNpx = minPx; 
      if(npx > maxPx) fNpx = maxPx; 
      Warning("SetNpx","Number of points must be >=%d && <= %d, fNpx set to %d",minPx,maxPx,fNpx);
   } 
   Update();
}


//______________________________________________________________________________
void TF1::SetParError(Int_t ipar, Double_t error)
{
   // Set error for parameter number ipar

   if (ipar < 0 || ipar > fNpar-1) return;
   fParErrors[ipar] = error;
}


//______________________________________________________________________________
void TF1::SetParErrors(const Double_t *errors)
{
   // Set errors for all active parameters
   // when calling this function, the array errors must have at least fNpar values

   if (!errors) return;
   for (Int_t i=0;i<fNpar;i++) fParErrors[i] = errors[i];
}


//______________________________________________________________________________
void TF1::SetParLimits(Int_t ipar, Double_t parmin, Double_t parmax)
{
   // Set limits for parameter ipar.
   //
   // The specified limits will be used in a fit operation
   // when the option "B" is specified (Bounds).
   // To fix a parameter, use TF1::FixParameter

   if (ipar < 0 || ipar > fNpar-1) return;
   Int_t i;
   if (!fParMin) {fParMin = new Double_t[fNpar]; for (i=0;i<fNpar;i++) fParMin[i]=0;}
   if (!fParMax) {fParMax = new Double_t[fNpar]; for (i=0;i<fNpar;i++) fParMax[i]=0;}
   fParMin[ipar] = parmin;
   fParMax[ipar] = parmax;
}


//______________________________________________________________________________
void TF1::SetRange(Double_t xmin, Double_t xmax)
{
   // Initialize the upper and lower bounds to draw the function.
   //
   // The function range is also used in an histogram fit operation
   // when the option "R" is specified.

   fXmin = xmin;
   fXmax = xmax;
   Update();
}


//______________________________________________________________________________
void TF1::SetSavedPoint(Int_t point, Double_t value)
{
   // Restore value of function saved at point

   if (!fSave) {
      fNsave = fNpx+3;
      fSave  = new Double_t[fNsave];
   }
   if (point < 0 || point >= fNsave) return;
   fSave[point] = value;
}


//______________________________________________________________________________
void TF1::SetTitle(const char *title)
{
   // Set function title
   //  if title has the form "fffffff;xxxx;yyyy", it is assumed that
   //  the function title is "fffffff" and "xxxx" and "yyyy" are the
   //  titles for the X and Y axis respectively.

   if (!title) return;
   fTitle = title;
   if (!fHistogram) return;
   fHistogram->SetTitle(title);
   if (gPad) gPad->Modified();
}


//______________________________________________________________________________
void TF1::Streamer(TBuffer &b)
{
   // Stream a class object.

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 4) {
         b.ReadClassBuffer(TF1::Class(), this, v, R__s, R__c);
         if (v == 5 && fNsave > 0) {
            //correct badly saved fSave in 3.00/06
            Int_t np = fNsave - 3;
            fSave[np]   = fSave[np-1];
            fSave[np+1] = fXmin;
            fSave[np+2] = fXmax;
         }
         return;
      }
      //====process old versions before automatic schema evolution
      TFormula::Streamer(b);
      TAttLine::Streamer(b);
      TAttFill::Streamer(b);
      TAttMarker::Streamer(b);
      if (v < 4) {
         Float_t xmin,xmax;
         b >> xmin; fXmin = xmin;
         b >> xmax; fXmax = xmax;
      } else {
         b >> fXmin;
         b >> fXmax;
      }
      b >> fNpx;
      b >> fType;
      b >> fChisquare;
      b.ReadArray(fParErrors);
      if (v > 1) {
         b.ReadArray(fParMin);
         b.ReadArray(fParMax);
      } else {
         fParMin = new Double_t[fNpar+1];
         fParMax = new Double_t[fNpar+1];
      }
      b >> fNpfits;
      if (v == 1) {
         b >> fHistogram;
         delete fHistogram; fHistogram = 0;
      }
      if (v > 1) {
         if (v < 4) {
            Float_t minimum,maximum;
            b >> minimum; fMinimum =minimum;
            b >> maximum; fMaximum =maximum;
         } else {
            b >> fMinimum;
            b >> fMaximum;
         }
      }
      if (v > 2) {
         b >> fNsave;
         if (fNsave > 0) {
            fSave = new Double_t[fNsave+10];
            b.ReadArray(fSave);
            //correct fSave limits to match new version
            fSave[fNsave]   = fSave[fNsave-1];
            fSave[fNsave+1] = fSave[fNsave+2];
            fSave[fNsave+2] = fSave[fNsave+3];
            fNsave += 3;
         } else fSave = 0;
      }
      b.CheckByteCount(R__s, R__c, TF1::IsA());
      //====end of old versions

   } else {
      Int_t saved = 0;
      if (fType > 0 && fNsave <= 0) { saved = 1; Save(fXmin,fXmax,0,0,0,0);}

      b.WriteClassBuffer(TF1::Class(),this);

      if (saved) {delete [] fSave; fSave = 0; fNsave = 0;}
   }
}


//______________________________________________________________________________
void TF1::Update()
{
   // Called by functions such as SetRange, SetNpx, SetParameters
   // to force the deletion of the associated histogram or Integral

   delete fHistogram;
   fHistogram = 0;
   if (fIntegral) {
      delete [] fIntegral; fIntegral = 0;
      delete [] fAlpha;    fAlpha    = 0;
      delete [] fBeta;     fBeta     = 0;
      delete [] fGamma;    fGamma    = 0;
   }
}


//______________________________________________________________________________
void TF1::RejectPoint(Bool_t reject)
{
   // Static function to set the global flag to reject points
   // the fgRejectPoint global flag is tested by all fit functions
   // if TRUE the point is not included in the fit.
   // This flag can be set by a user in a fitting function.
   // The fgRejectPoint flag is reset by the TH1 and TGraph fitting functions.

   fgRejectPoint = reject;
}


//______________________________________________________________________________
Bool_t TF1::RejectedPoint()
{
   // See TF1::RejectPoint above

   return fgRejectPoint;
}

//______________________________________________________________________________
Double_t TF1::Moment(Double_t n, Double_t a, Double_t b, const Double_t *params, Double_t epsilon)
{
   // Return nth moment of function between a and b
   //
   // See TF1::Integral() for parameter definitions

   // wrapped function in interface for integral calculation
   // using abs value of integral 

   TF1_EvalWrapper func(this, params, kTRUE, n); 

   ROOT::Math::GaussIntegrator giod;

   giod.SetFunction(func);
   giod.SetRelTolerance(epsilon);

   Double_t norm =  giod.Integral(a, b);
   if (norm == 0) {
      Error("Moment", "Integral zero over range");
      return 0;
   }

   // calculate now integral of x^n f(x)
   // wrapped the member function EvalNum in  interface required by integrator using the functor class 
   ROOT::Math::Functor1D xnfunc( &func, &TF1_EvalWrapper::EvalNMom);
   giod.SetFunction(xnfunc);

   Double_t res = giod.Integral(a,b)/norm;

   return res;
}


//______________________________________________________________________________
Double_t TF1::CentralMoment(Double_t n, Double_t a, Double_t b, const Double_t *params, Double_t epsilon)
{
   // Return nth central moment of function between a and b
   // (i.e the n-th moment around the mean value)   
   //
   // See TF1::Integral() for parameter definitions
   //   Author: Gene Van Buren <gene@bnl.gov>
  
   TF1_EvalWrapper func(this, params, kTRUE, n); 

   ROOT::Math::GaussIntegrator giod;

   giod.SetFunction(func);
   giod.SetRelTolerance(epsilon);

   Double_t norm =  giod.Integral(a, b);
   if (norm == 0) {
      Error("Moment", "Integral zero over range");
      return 0;
   }

   // calculate now integral of xf(x)
   // wrapped the member function EvalFirstMom in  interface required by integrator using the functor class 
   ROOT::Math::Functor1D xfunc( &func, &TF1_EvalWrapper::EvalFirstMom);
   giod.SetFunction(xfunc);

   // estimate of mean value
   Double_t xbar = giod.Integral(a,b)/norm;

   // use different mean value in function wrapper 
   func.fX0 = xbar; 
   ROOT::Math::Functor1D xnfunc( &func, &TF1_EvalWrapper::EvalNMom);
   giod.SetFunction(xnfunc);

   Double_t res = giod.Integral(a,b)/norm;
   return res;
}


//______________________________________________________________________________
// some useful static utility functions to compute sampling points for IntegralFast
//______________________________________________________________________________
#ifdef INTHEFUTURE
void TF1::CalcGaussLegendreSamplingPoints(TGraph *g, Double_t eps)
{
   // Type safe interface (static method)
   // The number of sampling points are taken from the TGraph

   if (!g) return;
   CalcGaussLegendreSamplingPoints(g->GetN(), g->GetX(), g->GetY(), eps);
}


//______________________________________________________________________________
TGraph *TF1::CalcGaussLegendreSamplingPoints(Int_t num, Double_t eps)
{
   // Type safe interface (static method)
   // A TGraph is created with new with num points and the pointer to the
   // graph is returned by the function. It is the responsibility of the
   // user to delete the object.
   // if num is invalid (<=0) NULL is returned

   if (num<=0)
      return 0;

   TGraph *g = new TGraph(num);
   CalcGaussLegendreSamplingPoints(g->GetN(), g->GetX(), g->GetY(), eps);
   return g;
}
#endif


//______________________________________________________________________________
void TF1::CalcGaussLegendreSamplingPoints(Int_t num, Double_t *x, Double_t *w, Double_t eps)
{
   // Type: unsafe but fast interface filling the arrays x and w (static method)
   //
   // Given the number of sampling points this routine fills the arrays x and w
   // of length num, containing the abscissa and weight of the Gauss-Legendre
   // n-point quadrature formula.
   //
   // Gauss-Legendre: W(x)=1  -1<x<1
   //                 (j+1)P_{j+1} = (2j+1)xP_j-jP_{j-1}
   //
   // num is the number of sampling points (>0)
   // x and w are arrays of size num
   // eps is the relative precision
   //
   // If num<=0 or eps<=0 no action is done.
   //
   // Reference: Numerical Recipes in C, Second Edition

   // This function is just kept like this for backward compatibility!

   ROOT::Math::GaussLegendreIntegrator gli(num,eps);
   gli.GetWeightVectors(x, w);


}
