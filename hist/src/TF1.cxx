// @(#)root/hist:$Name:  $:$Id: TF1.cxx,v 1.124 2006/04/26 06:10:23 brun Exp $
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
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TRandom.h"
#include "Api.h"
#include "TPluginManager.h"
#include "TVirtualUtilPad.h"
#include "TBrowser.h"
#include "TColor.h"
#include "TClass.h"

Bool_t TF1::fgAbsValue    = kFALSE;
Bool_t TF1::fgRejectPoint = kFALSE;
static TF1 *gHelper = 0;
static Double_t gErrorTF1 = 0;

ClassImp(TF1)

//______________________________________________________________________________
//
// a TF1 object is a 1-Dim function defined between a lower and upper limit.
// The function may be a simple function (see TFormula) or a precompiled
// user function.
// The function may have associated parameters.
// TF1 graphics function is via the TH1/TGraph drawing functions.
//
//  The following types of functions can be created:
//    A- Expression using variable x and no parameters
//    B- Expression using variable x with parameters
//    C- A general C function with parameters
//
//         +++++++++++++++++++++++++++++++++++
// ===>    + Example of a function of type A +
//         +++++++++++++++++++++++++++++++++++
//
//  Case A1 (inline expression using standard C++ functions/operators)
//  ------------------------------------------------------------------
//   TF1 *fa1 = new TF1("fa1","sin(x)/x",0,10);
//   fa1->Draw();
//Begin_Html
/*
<img src="gif/function1.gif">
*/
//End_Html
//
//  Case A2 (inline expression using TMath functions without parameters)
//  --------------------------------------------------------------------
//   TF1 *fa2 = new TF1("fa2","TMath::DiLog(x)",0,10);
//   fa2->Draw();
//
//  Case A3 (inline expression using a CINT function by name
//  --------------------------------------------------------
//   Double_t myFunc(x) {
//      return x+sin(x);
//   }
//   TF1 *fa3 = new TF1("fa4","myFunc(x)",-3,5);
//   fa3->Draw();
//
//
//         +++++++++++++++++++++++++++++++++++
// ===>    + Example of a function of type B+
//         +++++++++++++++++++++++++++++++++++
//
//  Case B1 (inline expression using standard C++ functions/operators)
//  ------------------------------------------------------------------
//  Example B1a
//  -----------
//   TF1 *fa = new TF1("fa","[0]*x*sin([1]*x)",-3,3);
//    This creates a function of variable x with 2 parameters.
//    The parameters must be initialized via:
//      fa->SetParameter(0,value_first_parameter);
//      fa->SetParameter(1,value_second_parameter);
//    Parameters may be given a name:
//      fa->SetParName(0,"Constant");
//
//  Example B1b
//  -----------
//   TF1 *fb = new TF1("fb","gaus(0)*expo(3)",0,10);
//     gaus(0) is a substitute for [0]*exp(-0.5*((x-[1])/[2])**2)
//        and (0) means start numbering parameters at 0
//     expo(3) is a substitute for exp([3]+[4]*x)
//
//  Case B2 (inline expression using TMath functions with parameters)
//  --------------------------------------------------------------------
//   TF1 *fb2 = new TF1("fa3","TMath::Landau(x,[0],[1],0)",-5,10);
//   fb2->SetParameters(0.2,1.3);
//   fb2->Draw();
//
//
//         +++++++++++++++++++++++++++++++++++
// ===>    + Example of a function of type C+
//         +++++++++++++++++++++++++++++++++++
//
//   Consider the macro myfunc.C below
//-------------macro myfunc.C-----------------------------
//Double_t myfunction(Double_t *x, Double_t *par)
//{
//   Float_t xx =x[0];
//   Double_t f = TMath::Abs(par[0]*sin(par[1]*xx)/xx);
//   return f;
//}
//void myfunc()
//{
//   TF1 *f1 = new TF1("myfunc",myfunction,0,10,2);
//   f1->SetParameters(2,1);
//   f1->SetParNames("constant","coefficient");
//   f1->Draw();
//}
//void myfit()
//{
//   TH1F *h1=new TH1F("h1","test",100,0,10);
//   h1->FillRandom("myfunc",20000);
//   TF1 *f1=gROOT->GetFunction("myfunc");
//   f1->SetParameters(800,1);
//   h1.Fit("myfunc");
//}
//--------end of macro myfunc.C---------------------------------
//
// In an interactive session you can do:
//   Root > .L myfunc.C
//   Root > myfunc();
//   Root > myfit();
//
//
//  TF1 objects can reference other TF1 objects (thanks John Odonnell)
//  of type A or B defined above.This excludes CINT interpreted functions
//  and compiled functions.
//  However, there is a restriction. A function cannot reference a basic
//  function if the basic function is a polynomial polN.
//  Example:
//{
//  TF1 *fcos = new TF1 ("fcos", "[0]*cos(x)", 0., 10.);
//  fcos->SetParNames( "cos");
//  fcos->SetParameter( 0, 1.1);
//
//  TF1 *fsin = new TF1 ("fsin", "[0]*sin(x)", 0., 10.);
//  fsin->SetParNames( "sin");
//  fsin->SetParameter( 0, 2.1);
//
//  TF1 *fsincos = new TF1 ("fsc", "fcos+fsin");
//
//  TF1 *fs2 = new TF1 ("fs2", "fsc+fsc");
//}
//
//     WHY TF1 CANNOT ACCEPT A CLASS MEMBER FUNCTION ?
//     ===============================================
// This is a frequently asked question.
// C++ is a strongly typed language. There is no way for TF1 (without
// recompiling this class) to know about all possible user defined data types.
// This also apply to the case of a static class function.
//
//------------------------------------------------------------------------

TF1 *TF1::fgCurrent = 0;

//______________________________________________________________________________
TF1::TF1(): TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*-*-*-*-*F1 default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   fXmin      = 0;
   fXmax      = 0;
   fNpx       = 100;
   fType      = 0;
   fNpfits    = 0;
   fNDF       = 0;
   fNsave     = 0;
   fChisquare = 0;
   fIntegral  = 0;
   fFunction  = 0;
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
   fFunction  = 0;
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
//*-*-*-*-*-*-*F1 constructor using name of an interpreted function*-*-*-*
//*-*          =======================================================
//*-*
//*-*  Creates a function of type C between xmin and xmax.
//*-*  name is the name of an interpreted CINT cunction.
//*-*  The function is defined with npar parameters
//*-*  fcn must be a function of type:
//*-*     Double_t fcn(Double_t *x, Double_t *params)
//*-*
//*-*  This constructor is called for functions of type C by CINT.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;
   fType       = 2;
   fFunction   = 0;
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
   if (f1old) delete f1old;
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
//*-*-*-*-*-*-*F1 constructor using pointer to an interpreted function*-*-*-*
//*-*          =======================================================
//*-*
//*-*  See TFormula constructor for explanation of the formula syntax.
//*-*
//*-*  Creates a function of type C between xmin and xmax.
//*-*  The function is defined with npar parameters
//*-*  fcn must be a function of type:
//*-*     Double_t fcn(Double_t *x, Double_t *params)
//*-*
//*-*  see tutorial; myfit for an example of use
//*-*  also test/stress.cxx (see function stress1)
//*-*
//*-*
//*-*  This constructor is called for functions of type C by CINT.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;
   fType       = 2;
   fFunction   = 0;
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
   if (f1old) delete f1old;
   SetName(name);

   if (gStyle) {
      SetLineColor(gStyle->GetFuncColor());
      SetLineWidth(gStyle->GetFuncWidth());
      SetLineStyle(gStyle->GetFuncStyle());
   }
   SetFillStyle(0);

   if (!fcn) return;
   char *funcname = G__p2f2funcname(fcn);
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
      Error("TF1","can not find any function at the address 0x%x. This function requested for %s",fcn,name);
   }
}

//______________________________________________________________________________
TF1::TF1(const char *name,Double_t (*fcn)(Double_t *, Double_t *), Double_t xmin, Double_t xmax, Int_t npar)
      :TFormula(), TAttLine(), TAttFill(), TAttMarker()
{
//*-*-*-*-*-*-*F1 constructor using a pointer to real function*-*-*-*-*-*-*-*
//*-*          ===============================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*   This constructor creates a function of type C when invoked
//*-*   with the normal C++ compiler.
//*-*
//*-*   see test program test/stress.cxx (function stress1) for an example.
//*-*   note the interface with an intermediate pointer.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;

   fType       = 1;
   fMethodCall = 0;
   fFunction   = fcn;

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
//*-*- Store formula in linked list of formula in ROOT

   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   if (f1old) delete f1old;
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
//*-*-*-*-*-*-*F1 constructor using a pointer to real function*-*-*-*-*-*-*-*
//*-*          ===============================================
//*-*
//*-*   npar is the number of free parameters used by the function
//*-*
//*-*   This constructor creates a function of type C when invoked
//*-*   with the normal C++ compiler.
//*-*
//*-*   see test program test/stress.cxx (function stress1) for an example.
//*-*   note the interface with an intermediate pointer.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fXmin       = xmin;
   fXmax       = xmax;
   fNpx        = 100;

   fType       = 1;
   fMethodCall = 0;
   typedef Double_t (*Function_t) (Double_t *, Double_t *);
   fFunction   = (Function_t)fcn;

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
//*-*- Store formula in linked list of formula in ROOT

   TF1 *f1old = (TF1*)gROOT->GetListOfFunctions()->FindObject(name);
   if (f1old) delete f1old;
   SetName(name);
   gROOT->GetListOfFunctions()->Add(this);

   if (!gStyle) return;
   SetLineColor(gStyle->GetFuncColor());
   SetLineWidth(gStyle->GetFuncWidth());
   SetLineStyle(gStyle->GetFuncStyle());
   SetFillStyle(0);

}

//______________________________________________________________________________
TF1& TF1::operator=(const TF1 &rhs) 
{
   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}

//______________________________________________________________________________
TF1::~TF1()
{
//*-*-*-*-*-*-*-*-*-*-*F1 default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================

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

   if (fParent) {
      if (fParent->InheritsFrom(TH1::Class())) {
         ((TH1*)fParent)->GetListOfFunctions()->Remove(this);
         return;
      }
      //parent may be a graph, or ??
      //The pad utility manager is required (a plugin)
      TVirtualUtilPad *util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
      if (!util) {
         TPluginHandler *h;
         if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilPad"))) {
            if (h->LoadPlugin() == -1)
               return;
            h->ExecPlugin(0);
            util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
         }
      }
      util->RemoveObject(fParent,this);
      fParent = 0;
   }
}

//______________________________________________________________________________
TF1::TF1(const TF1 &f1) : TFormula(), TAttLine(f1), TAttFill(f1), TAttMarker(f1)
{

   fXmin      = 0;
   fXmax      = 0;
   fNpx       = 100;
   fType      = 0;
   fNpfits    = 0;
   fNDF       = 0;
   fNsave     = 0;
   fChisquare = 0;
   fIntegral  = 0;
   fFunction  = 0;
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
   SetFillStyle(0);

   ((TF1&)f1).Copy(*this);
}

//______________________________________________________________________________
void TF1::AbsValue(Bool_t flag)
{
   // static function: set the fgAbsValue flag.
   // By default TF1::Integral uses the original function value to compute the integral
   // However, TF1::Moment, CentralMoment require to compute the integral
   // using the absolute value of the function.
   
   fgAbsValue = flag;
}

//______________________________________________________________________________
void TF1::Browse(TBrowser *b)
{
   Draw(b ? b->GetDrawOption() : "");
   gPad->Update();
}


//______________________________________________________________________________
void TF1::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this F1 to a new F1*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

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
   ((TF1&)obj).fFunction  = fFunction;
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
      TMethodCall *m = new TMethodCall();
      m->InitWithPrototype(fMethodCall->GetMethodName(),fMethodCall->GetProto());
      ((TF1&)obj).fMethodCall  = m;
   }
}

//______________________________________________________________________________
Double_t TF1::Derivative(Double_t x, Double_t *params, Double_t eps) const
{
  // returns the first derivative of the function at point x, 
  // computed by Richardson's extrapolation method (use 2 derivative estimates 
  // to compute a third, more accurate estimation)
  // first, derivatives with steps h and h/2 are computed by central difference formulas
  // D(h) = (f(x+h) - f(x-h))/2h
  // the final estimate D = (4*D(h/2) - D(h))/3
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
  // Getting the error via TF1::DerivativeError
  // -----------------
  //   (total error = roundoff error + interpolation error)
  // the estimate of the roundoff error is taken as follows:
  //    err = k*Sqrt(f(x)*f(x) + x*x*deriv*deriv)*Sqrt(Sum(ai)*(ai)),
  // where k is the double precision, ai are coefficients used in
  // central difference formulas
  // interpolation error is decreased by making the step size h smaller.
  //
  // Author: Anna Kreshuk
  
   const Double_t kC1 = 1e-15;

   if(eps< 1e-10 || eps > 1e-2) {
      Warning("Derivative","parameter esp=%g out of allowed range[1e-10,1e-2], reset to 0.001",eps);
      eps = 0.001;
   }
   Double_t xmin, xmax;
   GetRange(xmin, xmax);
   Double_t h = eps*(xmax-xmin);

   Double_t xx[1];
   TF1 *func = (TF1*)this;
   func->InitArgs(xx, params);
   xx[0] = x+h;     Double_t f1 = func->EvalPar(xx, params);
   xx[0] = x;       Double_t fx = func->EvalPar(xx, params);
   xx[0] = x-h;     Double_t f2 = func->EvalPar(xx, params);

   xx[0] = x+h/2;   Double_t g1 = func->EvalPar(xx, params);
   xx[0] = x-h/2;   Double_t g2 = func->EvalPar(xx, params);

   //compute the central differences
   Double_t h2    = 1/(2.*h);
   Double_t d0    = f1 - f2;
   Double_t d2    = 2*(g1 - g2);
   gErrorTF1       = kC1*h2*fx;  //compute the error
   Double_t deriv = h2*(4*d2 - d0)/3.;  
   return deriv;
}


//______________________________________________________________________________
Double_t TF1::Derivative2(Double_t x, Double_t *params, Double_t eps) const
{
   // returns the second derivative of the function at point x, 
   // computed by Richardson's extrapolation method (use 2 derivative estimates 
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //    D(h) = (f(x+h) - 2*f(x) + f(x-h))/(h*h)
   // the final estimate D = (4*D(h/2) - D(h))/3
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
   // Getting the error via TF1::DerivativeError
   // -----------------
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //    err = k*Sqrt(f(x)*f(x) + x*x*deriv*deriv)*Sqrt(Sum(ai)*(ai)),
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.
   //
   // Author: Anna Kreshuk

   const Double_t kC1 = 2*1e-15;

   if(eps< 1e-6 || eps > 1e-2) {
      Warning("Derivative2","parameter esp=%g out of allowed range[1e-6,1e-2], reset to 0.001",eps);
      eps = 0.001;
   }
   Double_t xmin, xmax;
   GetRange(xmin, xmax);
   Double_t h = eps*(xmax-xmin);

   Double_t xx[1];
   TF1 *func = (TF1*)this;
   func->InitArgs(xx, params);
   xx[0] = x+h;     Double_t f1 = func->EvalPar(xx, params);
   xx[0] = x;       Double_t f2 = func->EvalPar(xx, params);
   xx[0] = x-h;     Double_t f3 = func->EvalPar(xx, params);

   xx[0] = x+h/2;   Double_t g1 = func->EvalPar(xx, params);
   xx[0] = x-h/2;   Double_t g3 = func->EvalPar(xx, params);

   //compute the central differences
   Double_t hh    = 1/(h*h);
   Double_t d0    = f3 - 2*f2 + f1;
   Double_t d2    = 4*g3 - 8*f2 +4*g1;
   gErrorTF1       = kC1*hh*f2;  //compute the error
   Double_t deriv = hh*(4*d2 - d0)/3.;
   return deriv;
}


//______________________________________________________________________________
Double_t TF1::Derivative3(Double_t x, Double_t *params, Double_t eps) const
{
   // returns the third derivative of the function at point x, 
   // computed by Richardson's extrapolation method (use 2 derivative estimates 
   // to compute a third, more accurate estimation)
   // first, derivatives with steps h and h/2 are computed by central difference formulas
   //    D(h) = (f(x+2h) - 2*f(x+h) + 2*f(x-h) - f(x-2h))/(2*h*h*h)
   // the final estimate D = (4*D(h/2) - D(h))/3
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
   // Getting the error via TF1::DerivativeError
   // -----------------
   //   (total error = roundoff error + interpolation error)
   // the estimate of the roundoff error is taken as follows:
   //    err = k*Sqrt(f(x)*f(x) + x*x*deriv*deriv)*Sqrt(Sum(ai)*(ai)),
   // where k is the double precision, ai are coefficients used in
   // central difference formulas
   // interpolation error is decreased by making the step size h smaller.
   //
   // Author: Anna Kreshuk

   //const Double_t C1 = (1e-16)*TMath::Sqrt(5./2.)*TMath::Sqrt(16*64 + 1.)/3;
   const Double_t kC1 = 1e-15;

   if(eps< 1e-4 || eps > 1e-2) {
      Warning("Derivative3","parameter esp=%g out of allowed range[1e-4,1e-2], reset to 0.001",eps);
      eps = 0.001;
   }
   Double_t xmin, xmax;
   GetRange(xmin, xmax);
   Double_t h = eps*(xmax-xmin);

   Double_t xx[1];
   TF1 *func = (TF1*)this;
   func->InitArgs(xx, params);
   xx[0] = x+2*h;   Double_t f1 = func->EvalPar(xx, params);
   xx[0] = x+h;     Double_t f2 = func->EvalPar(xx, params);
   xx[0] = x-h;     Double_t f3 = func->EvalPar(xx, params);
   xx[0] = x-2*h;   Double_t f4 = func->EvalPar(xx, params);
   xx[0] = x;       Double_t fx = func->EvalPar(xx, params);
   xx[0] = x+h/2;   Double_t g2 = func->EvalPar(xx, params);
   xx[0] = x-h/2;   Double_t g3 = func->EvalPar(xx, params);

   //compute the central differences
   Double_t hhh  = 1/(h*h*h);
   Double_t d0   = 0.5*f1 - f2 +f3 - 0.5*f4;
   Double_t d2   = 4*f2 - 8*g2 +8*g3 - 4*f3;
   gErrorTF1      = kC1*hhh*fx;   //compute the error
   Double_t deriv = hhh*(4*d2 - d0)/3.;
   return deriv;
}

//______________________________________________________________________________
Double_t TF1::DerivativeError()
{
   //static function returning the error of the last call to the Derivative functions
   
   return gErrorTF1;
}

//______________________________________________________________________________
Int_t TF1::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a function*-*-*-*-*
//*-*                  ===============================================
//*-*  Compute the closest distance of approach from point px,py to this function.
//*-*  The distance is computed in pixels units.
//*-*
//*-*  Note that px is called with a negative value when the TF1 is in
//*-*  TGraph or TH1 list of functions. In this case there is no point
//*-*  looking at the histogram axis.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

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
//*-*-*-*-*-*-*-*-*-*-*Draw this function with its current attributes*-*-*-*-*
//*-*                  ==============================================
//*-*
//*-* Possible option values are:
//*-*   "SAME"  superimpose on top of existing picture
//*-*   "L"     connect all computed points with a straight line
//*-*   "C"     connect all computed points with a smooth curve.
//*-*   "FC"    draw a fill area below a smooth curve
//*-*
//*-* Note that the default value is "L". Therefore to draw on top
//*-* of an existing picture, specify option "LSAME"
//*-*
//*-* NB. You must use DrawCopy if you want to draw several times the same
//*-*     function in the current canvas.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TString opt = option;
   opt.ToLower();
   if (gPad && !opt.Contains("same")) gPad->Clear();

   AppendPad(option);
}

//______________________________________________________________________________
TF1 *TF1::DrawCopy(Option_t *option) const
{
//*-*-*-*-*-*-*-*Draw a copy of this function with its current attributes*-*-*
//*-*            ========================================================
//*-*
//*-*  This function MUST be used instead of Draw when you want to draw
//*-*  the same function with different parameters settings in the same canvas.
//*-*
//*-* Possible option values are:
//*-*   "SAME"  superimpose on top of existing picture
//*-*   "L"     connect all computed points with a straight line
//*-*   "C"     connect all computed points with a smooth curve.
//*-*   "FC"    draw a fill area below a smooth curve
//*-*
//*-* Note that the default value is "L". Therefore to draw on top
//*-* of an existing picture, specify option "LSAME"
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   TF1 *newf1 = new TF1();
   Copy(*newf1);
   newf1->AppendPad(option);
   newf1->SetBit(kCanDelete);
   return newf1;
}

//______________________________________________________________________________
void TF1::DrawDerivative(Option_t *option)
{
// Draw derivative of this function
//
// An intermediate TGraph object is built and drawn with option.
//
// The resulting graph will be drawn into the current pad.
// If this function is used via the context menu, it recommended
// to create a new canvas/pad before invoking this function.
   
   TVirtualPad *pad = gROOT->GetSelectedPad();
   TVirtualPad *padsav = gPad;
   if (pad) pad->cd();

   char cmd[512];
   sprintf(cmd,"{TGraph *R__%s_Derivative = new TGraph((TF1*)0x%lx,\"d\");R__%s_Derivative->Draw(\"%s\");}",GetName(),(Long_t)this,GetName(),option);
   gROOT->ProcessLine(cmd);
   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TF1::DrawIntegral(Option_t *option)
{
// Draw integral of this function
//
// An intermediate TGraph object is built and drawn with option.
//
// The resulting graph will be drawn into the current pad.
// If this function is used via the context menu, it recommended
// to create a new canvas/pad before invoking this function.

   TVirtualPad *pad = gROOT->GetSelectedPad();
   TVirtualPad *padsav = gPad;
   if (pad) pad->cd();

   char cmd[512];
   sprintf(cmd,"{TGraph *R__%s_Integral = new TGraph((TF1*)0x%lx,\"i\");R__%s_Integral->Draw(\"%s\");}",GetName(),(Long_t)this,GetName(),option);
   gROOT->ProcessLine(cmd);
   if (padsav) padsav->cd();
}

//______________________________________________________________________________
void TF1::DrawF1(const char *formula, Double_t xmin, Double_t xmax, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*Draw formula between xmin and xmax*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==================================
//*-*

   if (Compile(formula)) return;

   SetRange(xmin, xmax);

   Draw(option);
}

//______________________________________________________________________________
void TF1::DrawPanel()
{
//*-*-*-*-*-*-*Display a panel with all function drawing options*-*-*-*-*-*
//*-*          =================================================
//*-*
//*-*   See class TDrawPanelHist for example

   //The pad utility manager is required (a plugin)
   TVirtualUtilPad *util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
   if (!util) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualUtilPad"))) {
         if (h->LoadPlugin() == -1)
            return;
         h->ExecPlugin(0);
         util = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
      }
   }
   util->DrawPanel(gPad,this);
}

//______________________________________________________________________________
Double_t TF1::Eval(Double_t x, Double_t y, Double_t z, Double_t t) const
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
//*-*
//*-*   Computes the value of this function (general case for a 3-d function)
//*-*   at point x,y,z.
//*-*   For a 1-d function give y=0 and z=0
//*-*   The current value of variables x,y,z is passed through x, y and z.
//*-*   The parameters used will be the ones in the array params if params is given
//*-*    otherwise parameters will be taken from the stored data members fParams
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

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
//*-*-*-*-*-*Evaluate function with given coordinates and parameters*-*-*-*-*-*
//*-*        =======================================================
//*-*
//      Compute the value of this function at point defined by array x
//      and current values of parameters in array params.
//      If argument params is omitted or equal 0, the internal values
//      of parameters (array fParams) will be used instead.
//      For a 1-D function only x[0] must be given.
//      In case of a multi-dimemsional function, the arrays x must be
//      filled with the corresponding number of dimensions.
//
//   WARNING. In case of an interpreted function (fType=2), it is the
//   user's responsability to initialize the parameters via InitArgs
//   before calling this function.
//   InitArgs should be called at least once to specify the addresses
//   of the arguments x and params.
//   InitArgs should be called everytime these addresses change.
//

   fgCurrent = this;
   
   if (fType == 0) return TFormula::EvalPar(x,params);
   Double_t result = 0;
   if (fType == 1)  {
      if (fFunction) {
         if (params) result = (*fFunction)((Double_t*)x,(Double_t*)params);
         else        result = (*fFunction)((Double_t*)x,fParams);
      }else          result = GetSave(x);
   }
   if (fType == 2) {
      if (fMethodCall) fMethodCall->Execute(result);
      else             result = GetSave(x);
   }
   return result;
}

//______________________________________________________________________________
void TF1::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function is called when a F1 is clicked with the locator
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fHistogram->ExecuteEvent(event,px,py);

   if (!gPad->GetView()) {
      if (event == kMouseMotion)  gPad->SetCursor(kHand);
   }
}

//______________________________________________________________________________
void TF1::FixParameter(Int_t ipar, Double_t value)
{
// Fix the value of a parameter
//     The specified value will be used in a fit operation

   if (ipar < 0 || ipar > fNpar-1) return;
   SetParameter(ipar,value);
   if (value != 0) SetParLimits(ipar,value,value);
   else            SetParLimits(ipar,1,1);
}


//______________________________________________________________________________
TF1 *TF1::GetCurrent()
{
// static function returning the current function being processed
   return fgCurrent;
}


//______________________________________________________________________________
TH1 *TF1::GetHistogram() const
{
// return a pointer to the histogram used to vusualize the function

   if (fHistogram) return fHistogram;

   // may be function has not yet be painted. force a pad update
   //gPad->Modified();
   //gPad->Update();
   ((TF1*)this)->Paint();
   return fHistogram;
}

//______________________________________________________________________________
Double_t TF1::GetMaximum(Double_t xmin, Double_t xmax) const
{
// return the maximum value of the function
// Method:
//  First, the grid search is used to bracket the maximum 
//  with the step size = (xmax-xmin)/fNpx. This way, the step size
//  can be controlled via the SetNpx() function. If the function is
//  unimodal or if its extrema are far apart, setting the fNpx to 
//  a small value speeds the algorithm up many times.  
//  Then, Brent's method is applied on the bracketed interval

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}
   Double_t x;
   Int_t niter=0;
   x = MinimStep(3, xmin, xmax, 0);
   Bool_t ok = kTRUE;
   x = MinimBrent(3, xmin, xmax, x, 0, ok);
   while (!ok){
      if (niter>10){
         Error("GetMaximum", "maximum search didn't converge");
         break;
      }
      x=MinimStep(3, xmin, xmax,0);
      x = MinimBrent(3, xmin, xmax, x, 0, ok);
      niter++;
   }
   return x;
}

//______________________________________________________________________________
Double_t TF1::GetMaximumX(Double_t xmin, Double_t xmax) const
{
// return the X value corresponding to the maximum value of the function
// Method:
//  First, the grid search is used to bracket the maximum 
//  with the step size = (xmax-xmin)/fNpx. This way, the step size
//  can be controlled via the SetNpx() function. If the function is
//  unimodal or if its extrema are far apart, setting the fNpx to 
//  a small value speeds the algorithm up many times.  
//  Then, Brent's method is applied on the bracketed interval

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}
   Double_t x;
   Int_t niter=0;
   x = MinimStep(2, xmin, xmax, 0);
   Bool_t ok = kTRUE;
   x = MinimBrent(2, xmin, xmax, x, 0, ok);
   while (!ok){
      if (niter>10){
         Error("GetMaximumX", "maximum search didn't converge");
         break;
      }
      x=MinimStep(2, xmin, xmax, 0);
      x = MinimBrent(2, xmin, xmax, x, 0, ok);
      niter++;
   }
   return x;
}

//______________________________________________________________________________
Double_t TF1::GetMinimum(Double_t xmin, Double_t xmax) const
{
// Returns the minimum value of the function on the (xmin, xmax) interval
// Method:
//  First, the grid search is used to bracket the maximum 
//  with the step size = (xmax-xmin)/fNpx. This way, the step size
//  can be controlled via the SetNpx() function. If the function is
//  unimodal or if its extrema are far apart, setting the fNpx to 
//  a small value speeds the algorithm up many times.  
//  Then, Brent's method is applied on the bracketed interval

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}
   Double_t x;
   Int_t niter=0;
   x = MinimStep(1, xmin, xmax, 0);
   Bool_t ok = kTRUE;
   x = MinimBrent(1, xmin, xmax, x, 0, ok);
   while (!ok){
      if (niter>10){
         Error("GetMinimum", "minimum search didn't converge");
         break;
      }
      x=MinimStep(1, xmin, xmax,0);
      x = MinimBrent(1, xmin, xmax, x, 0, ok);
      niter++;
   }
   return x;
}

//______________________________________________________________________________
Double_t TF1::GetMinimumX(Double_t xmin, Double_t xmax) const
{
// Returns the X value corresponding to the minimum value of the function on the
// (xmin, xmax) interval
// Method:
//  First, the grid search is used to bracket the maximum 
//  with the step size = (xmax-xmin)/fNpx. This way, the step size
//  can be controlled via the SetNpx() function. If the function is
//  unimodal or if its extrema are far apart, setting the fNpx to 
//  a small value speeds the algorithm up many times.  
//  Then, Brent's method is applied on the bracketed interval

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}
   Int_t niter=0;
   Double_t x;

   x = MinimStep(0, xmin, xmax, 0);
   Bool_t ok = kTRUE;
   x = MinimBrent(0, xmin, xmax, x, 0, ok);
   while (!ok){
      if (niter>10){
         Error("GetMinimumX", "minimum search didn't converge");
         break;
      }
      x=MinimStep(0, xmin, xmax,0);
      x = MinimBrent(0, xmin, xmax, x, 0, ok);
      niter++;
   }
   return x;
}

//______________________________________________________________________________
Double_t TF1::GetX(Double_t fy, Double_t xmin, Double_t xmax) const
{
// Returns the X value corresponding to the function value fy for (xmin<x<xmax).
// Method:
//  First, the grid search is used to bracket the maximum 
//  with the step size = (xmax-xmin)/fNpx. This way, the step size
//  can be controlled via the SetNpx() function. If the function is
//  unimodal or if its extrema are far apart, setting the fNpx to 
//  a small value speeds the algorithm up many times.  
//  Then, Brent's method is applied on the bracketed interval

   if (xmin >= xmax) {xmin = fXmin; xmax = fXmax;}
   Int_t niter=0;
   Double_t x;
   x = MinimStep(4, xmin, xmax, fy);
   Bool_t ok = kTRUE;
   x = MinimBrent(4, xmin, xmax, x, fy, ok);
   while (!ok){
      if (niter>10){
         Error("GetX", "Search didn't converge");
         break;
      }
      x=MinimStep(4, xmin, xmax, fy);
      x = MinimBrent(4, xmin, xmax, x, fy, ok);
      niter++;
   }
   return x;
}

//______________________________________________________________________________
Double_t TF1::MinimStep(Int_t type, Double_t &xmin, Double_t &xmax, Double_t fy) const
{
//   Grid search implementation, used to bracket the minimum and later
//   use Brent's method with the bracketed interval
//   The step of the search is set to (xmax-xmin)/fNpx
//   type: 0-returns MinimumX
//         1-returns Minimum
//         2-returns MaximumX
//         3-returns Maximum
//         4-returns X corresponding to fy

   Double_t x,y, dx;
   dx = (xmax-xmin)/(fNpx-1);
   Double_t xxmin = xmin;
   Double_t yymin;
   if (type < 2)
      yymin = Eval(xmin);
   else if (type < 4)
      yymin = -Eval(xmin);
   else
      yymin = TMath::Abs(Eval(xmin)-fy);

   for (Int_t i=1; i<=fNpx-1; i++) {
      x = xmin + i*dx;
      if (type < 2)
         y = Eval(x);
      else if (type < 4)
         y = -Eval(x);
      else
         y = TMath::Abs(Eval(x)-fy);
      if (y < yymin) {xxmin = x; yymin = y;}
   }

   xmin = TMath::Max(xmin,xxmin-dx);
   xmax = TMath::Min(xmax,xxmin+dx);

   return TMath::Min(xxmin, xmax);
}


//______________________________________________________________________________
Double_t TF1::MinimBrent(Int_t type, Double_t &xmin, Double_t &xmax, Double_t xmiddle, Double_t fy, Bool_t &ok) const
{
   //Finds a minimum of a function, if the function is unimodal  between xmin and xmax
   //This method uses a combination of golden section search and parabolic interpolation
   //Details about convergence and properties of this algorithm can be
   //found in the book by R.P.Brent "Algorithms for Minimization Without Derivatives"
   //or in the "Numerical Recipes", chapter 10.2
   //
   //type: 0-returns MinimumX
   //      1-returns Minimum
   //      2-returns MaximumX
   //      3-returns Maximum
   //      4-returns X corresponding to fy
   //if ok=true the method has converged

   Double_t eps = 1e-10;
   Double_t t = 1e-8;
   Int_t itermax = 100;

   Double_t c = (3.-TMath::Sqrt(5.))/2.; //comes from golden section
   Double_t u, v, w, x, fv, fu, fw, fx, e, p, q, r, t2, d=0, m, tol;
   v = w = x = xmiddle;
   e=0;

   Double_t a=xmin;
   Double_t b=xmax;
   if (type < 2)
      fv = fw = fx = Eval(x);
   else if (type < 4)
      fv = fw = fx = -Eval(x);
   else
      fv = fw = fx = TMath::Abs(Eval(x)-fy);

   for (Int_t i=0; i<itermax; i++){
      m=0.5*(a + b);
      tol = eps*(TMath::Abs(x))+t;
      t2 = 2*tol;
      if (TMath::Abs(x-m) <= (t2-0.5*(b-a))) {
         //converged, return x
         ok=kTRUE;
         if (type==1)
            return fx;
         else if (type==3)
            return -fx;
         else
            return x;
      }

      if (TMath::Abs(e)>tol){
         //fit parabola
         r = (x-w)*(fx-fv);
         q = (x-v)*(fx-fw);
         p = (x-v)*q - (x-w)*r;
         q = 2*(q-r);
         if (q>0) p=-p;
         else q=-q;
         r=e;
         e=d;
           
         if (TMath::Abs(p) < TMath::Abs(0.5*q*r) || p < q*(a-x) || p < q*(b-x)) {
            //a parabolic interpolation step
            d = p/q;
            u = x+d;
            if (u-a < t2 || b-u < t2)
               d=TMath::Sign(tol, m-x);
         } else {
            e=(x>=m ? a-x : b-x);
            d = c*e;
         }
      } else {
         e=(x>=m ? a-x : b-x);
         d = c*e;
      }
      u = (TMath::Abs(d)>=tol ? x+d : x+TMath::Sign(tol, d));
      if (type < 2)
         fu = Eval(u);
      else if (type < 4)
         fu = -Eval(u);
      else
         fu = TMath::Abs(Eval(u)-fy);
      //update a, b, v, w and x
      if (fu<=fx){
         if (u<x) b=x;
         else a=x;
         v=w; fv=fw; w=x; fw=fx; x=u; fx=fu;
      } else {
         if (u<x) a=u;
         else b=u;
         if (fu<=fw || w==x){
            v=w; fv=fw; w=u; fw=fu;
         }
         else if (fu<=fv || v==x || v==w){
            v=u; fv=fu;
         }
      }
   }
   //didn't converge
   ok = kFALSE;
   xmin = a;
   xmax = b;
   return x;
}

//______________________________________________________________________________
Int_t TF1::GetNDF() const
{
// return the number of degrees of freedom in the fit
// the fNDF parameter has been previously computed during a fit.
// The number of degrees of freedom corresponds to the number of points
// used in the fit minus the number of free parameters.

   if (fNDF == 0) return fNpfits-fNpar;
   return fNDF;
}

//______________________________________________________________________________
Int_t TF1::GetNumberFreeParameters() const
{
// return the number of free parameters
   
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
//   Redefines TObject::GetObjectInfo.
//   Displays the function info (x, function value
//   corresponding to cursor position px,py
//
   static char info[64];
   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   sprintf(info,"(x=%g, f=%g)",x,((TF1*)this)->Eval(x));
   return info;
}

//______________________________________________________________________________
Double_t TF1::GetParError(Int_t ipar) const
{
   //return value of parameter number ipar

   if (ipar < 0 || ipar > fNpar-1) return 0;
   return fParErrors[ipar];
}

//______________________________________________________________________________
void TF1::GetParLimits(Int_t ipar, Double_t &parmin, Double_t &parmax) const
{
//*-*-*-*-*-*Return limits for parameter ipar*-*-*-*
//*-*        ================================

   parmin = 0;
   parmax = 0;
   if (ipar < 0 || ipar > fNpar-1) return;
   if (fParMin) parmin = fParMin[ipar];
   if (fParMax) parmax = fParMax[ipar];
}

//______________________________________________________________________________
Double_t TF1::GetProb() const
{
// return the fit probability
   
   if (fNDF <= 0) return 0;
   return TMath::Prob(fChisquare,fNDF);
}
   
//______________________________________________________________________________
Int_t TF1::GetQuantiles(Int_t nprobSum, Double_t *q, const Double_t *probSum)
{
//  Compute Quantiles for density distribution of this function
//     Quantile x_q of a probability distribution Function F is defined as
//
//        F(x_q) = Integral_{xmin}^(x_q) f dx = q with 0 <= q <= 1.
//
//     For instance the median x_0.5 of a distribution is defined as that value
//     of the random variable for which the distribution function equals 0.5:
//
//        F(x_0.5) = Probability(x < x_0.5) = 0.5
//
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

   const Int_t npx     = TMath::Min(250,TMath::Max(50,2*nprobSum));
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
      Int_t bin  = TMath::Max(TMath::BinarySearch(npx+1,integral.GetArray(),r)-1,(Long64_t)0);
      while (bin < npx-1 && integral[bin+1] == r) {
         if (integral[bin+2] == r) bin++;
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
//*-*
//*-*   The distribution contained in the function fname (TF1) is integrated
//*-*   over the channel contents.
//*-*   It is normalized to 1.
//*-*   For each bin the integral is approximated by a parabola.
//*-*   The parabola coefficients are stored as non persistent data members
//*-*   Getting one random number implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which bin in the normalized integral r1 corresponds to
//*-*     - Evaluate the parabolic curve in the selected bin to find
//*-*       the corresponding X value.
//*-*   The parabolic approximation is very good as soon as the number
//*-*   of bins is greater than 50.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

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
   Double_t r  = gRandom->Rndm();
   Int_t bin  = TMath::BinarySearch(fNpx,fIntegral,r);
   Double_t rr = r - fIntegral[bin];

   Double_t xx;
   if(fGamma[bin] != 0)
      xx = (-fBeta[bin] + TMath::Sqrt(fBeta[bin]*fBeta[bin]+2*fGamma[bin]*rr))/fGamma[bin];
   else 
      xx = rr/fBeta[bin];
   Double_t x = fAlpha[bin] + xx;
   return x;
}

//______________________________________________________________________________
Double_t TF1::GetRandom(Double_t xmin, Double_t xmax)
{
// Return a random number following this function shape in [xmin,xmax]
//*-*
//*-*   The distribution contained in the function fname (TF1) is integrated
//*-*   over the channel contents.
//*-*   It is normalized to 1.
//*-*   For each bin the integral is approximated by a parabola.
//*-*   The parabola coefficients are stored as non persistent data members
//*-*   Getting one random number implies:
//*-*     - Generating a random number between 0 and 1 (say r1)
//*-*     - Look in which bin in the normalized integral r1 corresponds to
//*-*     - Evaluate the parabolic curve in the selected bin to find
//*-*       the corresponding X value.
//*-*   The parabolic approximation is very good as soon as the number
//*-*   of bins is greater than 50.
//*-*
//*-*  IMPORTANT NOTE
//*-*  The integral of the function is computed at fNpx points. If the function
//*-*  has sharp peaks, you should increase the number of points (SetNpx)
//*-*  such that the peak is correctly tabulated at several points.
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*

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
//*-*-*-*-*-*-*-*-*-*-*Return range of a 1-D function*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================

   xmin = fXmin;
   xmax = fXmax;
}

//______________________________________________________________________________
void TF1::GetRange(Double_t &xmin, Double_t &ymin,  Double_t &xmax, Double_t &ymax) const
{
//*-*-*-*-*-*-*-*-*-*-*Return range of a 2-D function*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==============================

   xmin = fXmin;
   xmax = fXmax;
   ymin = 0;
   ymax = 0;
}

//______________________________________________________________________________
void TF1::GetRange(Double_t &xmin, Double_t &ymin, Double_t &zmin, Double_t &xmax, Double_t &ymax, Double_t &zmax) const
{
//*-*-*-*-*-*-*-*-*-*-*Return range of function*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

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
   Int_t np = fNsave - 3;
   Double_t xmin = Double_t(fSave[np+1]);
   Double_t xmax = Double_t(fSave[np+2]);
   Double_t x    = Double_t(xx[0]);
   Double_t dx   = (xmax-xmin)/np;
   if (x < xmin || x > xmax) return 0;
   if (dx <= 0) return 0;

   Int_t bin     = Int_t((x-xmin)/dx);
   Double_t xlow = xmin + bin*dx;
   Double_t xup  = xlow + dx;
   Double_t ylow = fSave[bin];
   Double_t yup  = fSave[bin+1];
   Double_t y    = ((xup*ylow-xlow*yup) + x*(yup-ylow))/dx;
   return y;
}

//______________________________________________________________________________
TAxis *TF1::GetXaxis() const
{
   // Get x axis of the function.

   //if (!gPad) return 0;
   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetXaxis();
}

//______________________________________________________________________________
TAxis *TF1::GetYaxis() const
{
   // Get y axis of the function.

   //if (!gPad) return 0;
   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetYaxis();
}

//______________________________________________________________________________
TAxis *TF1::GetZaxis() const
{
   // Get z axis of the function. (In case this object is a TF2 or TF3)

   //if (!gPad) return 0;
   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetZaxis();
}

//______________________________________________________________________________
void TF1::GradientPar(const Double_t *x, Double_t *grad, Double_t eps)
{
   //Compute the gradient wrt parameters
   //Parameters:
   //x - point, were the gradient is computed
   //grad - used to return the computed gradient, assumed to be of at least fNpar size
   //eps - if the errors of parameters have been computed, the step used in
   //numerical differentiation is eps*parameter_error.
   //if the errors have not been computed, step=eps is used
   //default value of eps = 0.01
   //Method is the same as in Derivative() function
   
   if(eps< 1e-10 || eps > 1) {
      Warning("Derivative","parameter esp=%g out of allowed range[1e-10,1], reset to 0.01",eps);
      eps = 0.01;
   }
   Double_t h;
   TF1 *func = (TF1*)this;
   //save original parameters
   Double_t *params=0;
   Double_t par_local[20];
   Bool_t isAllocated=kFALSE;
   if (fNpar > 20){
      params = new Double_t[fNpar];
      isAllocated = kTRUE;
   } else
      params = par_local;

   Bool_t errorsComputed=kFALSE;
   for (Int_t ipar=0; ipar<fNpar; ipar++){
      params[ipar]=fParams[ipar];
      if (func->GetParError(ipar)!=0)
         errorsComputed=kTRUE;
   }

   for (Int_t ipar=0; ipar<fNpar; ipar++){

      func->InitArgs(x, params);
      if (errorsComputed)
         h = eps*func->GetParError(ipar);
      else 
         h=eps;
      params[ipar] = fParams[ipar]+h;     Double_t f1 = func->EvalPar(x,params);
      params[ipar] = fParams[ipar]-h;     Double_t f2 = func->EvalPar(x,params);
      
      params[ipar] = fParams[ipar]+h/2;   Double_t g1 = func->EvalPar(x,params);
      params[ipar] = fParams[ipar]-h/2;   Double_t g2 = func->EvalPar(x,params);
      
      //compute the central differences
      Double_t h2    = 1/(2.*h);
      Double_t d0    = f1 - f2;
      Double_t d2    = 2*(g1 - g2);
      
      grad[ipar] = h2*(4*d2 - d0)/3.;
      params[ipar]=fParams[ipar];  
   }
   if (isAllocated)
      delete [] params;
}

//______________________________________________________________________________
void TF1::InitArgs(const Double_t *x, const Double_t *params)
{
//*-*-*-*-*-*-*-*-*-*-*Initialize parameters addresses*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================

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
//     Create the basic function objects

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
//*-*-*-*-*-*-*-*-*Return Integral of function between a and b*-*-*-*-*-*-*-*
//
//   based on original CERNLIB routine DGAUSS by Sigfried Kolbig
//   converted to C++ by Rene Brun
//
//Begin_Html
/*
<BR>
<P>
This function computes,
to an attempted specified accuracy, the value of the integral
 <P> <IMG WIDTH=140 HEIGHT=45 ALIGN=BOTTOM ALT="displaymath120" SRC="gif/gaus_img1.gif"  > <P>

<p><b>Usage:</b><p>
In any arithmetic expression, this function
has the approximate value of the integral I.
<DL COMPACT>
<DT>a,b
<DD>End-points of integration interval. Note that <TT>B</TT>
may be less than <TT>A</TT>.
<DT>params
<DD>Array of function parameters. If 0, use current parameters.
<DT>epsilon
<DD>Accuracy parameter (see <B>Accuracy</B>).
</DL>
<P>
<p><b>Method:</b><p>
For any interval [<I>a</I>,<I>b</I>] we define  <IMG WIDTH=49 HEIGHT=26 ALIGN=MIDDLE ALT="tex2html_wrap_inline128" SRC="gif/gaus_img3.gif"  >  and  <IMG WIDTH=55 HEIGHT=26 ALIGN=MIDDLE ALT="tex2html_wrap_inline130" SRC="gif/gaus_img4.gif"  >  to be the
8-point and 16-point Gaussian quadrature approximations to
     <P> <IMG WIDTH=120 HEIGHT=40 ALIGN=BOTTOM ALT="displaymath132" SRC="gif/gaus_img5.gif"  > <P>
and define
<P> <IMG WIDTH=220 HEIGHT=57 ALIGN=BOTTOM ALT="displaymath134" SRC="gif/gaus_img6.gif"  > <P>
Then,
<P> <IMG WIDTH=180 HEIGHT=57 ALIGN=BOTTOM ALT="displaymath138" SRC="gif/gaus_img7.gif"  > <P>
<P>
where, starting with  <IMG WIDTH=48 HEIGHT=22 ALIGN=MIDDLE ALT="tex2html_wrap_inline140" SRC="gif/gaus_img8.gif"  >  and finishing with  <IMG WIDTH=49 HEIGHT=22 ALIGN=MIDDLE ALT="tex2html_wrap_inline142" SRC="gif/gaus_img9.gif"  > ,
the subdivision points  <IMG WIDTH=107 HEIGHT=26 ALIGN=MIDDLE ALT="tex2html_wrap_inline144" SRC="gif/gaus_img10.gif"  >  are given by
<P> <IMG WIDTH=180 HEIGHT=40 ALIGN=BOTTOM ALT="displaymath146" SRC="gif/gaus_img11.gif"  > <P>
with  <IMG WIDTH=8 HEIGHT=11 ALIGN=BOTTOM ALT="tex2html_wrap_inline148" SRC="gif/gaus_img12.gif"  >  equal to the first member of the sequence
 <IMG WIDTH=66 HEIGHT=26 ALIGN=MIDDLE ALT="tex2html_wrap_inline150" SRC="gif/gaus_img13.gif"  >  for which
 <IMG WIDTH=117 HEIGHT=26 ALIGN=MIDDLE ALT="tex2html_wrap_inline152" SRC="gif/gaus_img14.gif"  > .
If, at any stage in the process of subdivision, the ratio
  <P> <IMG WIDTH=180 HEIGHT=42 ALIGN=BOTTOM ALT="displaymath154" SRC="gif/gaus_img15.gif"  > <P>
is so small that 1+0.005<I>q</I> is indistinguishable from 1 to
machine accuracy, an error exit occurs with the function value
set equal to zero.
<P>
<p><b>Accuracy:</b><p>
Unless there is severe cancellation of positive and negative
values of <I>f</I>(<I>x</I>) over the interval [<I>A</I>,<I>B</I>], the argument <TT>EPS</TT>
may be considered as specifying a bound on the <I>relative</I> error of
<I>I</I> in the case  |<I>I</I>|&gt;1, and a bound on the <I>absolute</I> error in
the case |<I>I</I>|&lt;1. More precisely, if <I>k</I> is the number of sub-intervals
contributing to the approximation (see Method), and if
<P> <IMG WIDTH=180 HEIGHT=50 ALIGN=BOTTOM ALT="displaymath170" SRC="gif/gaus_img16.gif"  > <P>
then the relation
<P> <IMG WIDTH=140 HEIGHT=50 ALIGN=BOTTOM ALT="displaymath172" SRC="gif/gaus_img17.gif"  > <P>
will nearly always be true, provided the routine terminates
without printing an error message. For functions
<I>f</I> having no singularities in the closed interval [<I>A</I>,<I>B</I>]
the accuracy will usually be much higher than this.
<P>
<p><b>Error handling:</b><p>
The requested accuracy cannot be
obtained (see <B>Method</B>).
The function value is set equal to zero.
<P>
<p><b>Note1:</b><p>
Values of the function <I>f</I>(<I>x</I>) at the interval end-points
<I>A</I> and <I>B</I> are not required. The subprogram may therefore
be used when these values are undefined.
<BR><HR>
<p><b>Note2:</b><p>
Instead of TF1::Integral, you may want to use the combination of
<b>TF1::CalcGaussLegendreSamplingPoints</b> and <b>TF1::IntegralFast</b>.
See an example with the following script:
<pre>
void gint() {
   TF1 *g = new TF1("g","gaus",-5,5);
   g->SetParameters(1,0,1);
   //default gaus integration method uses 6 points
   //not suitable to integrate on a large domain
   double r1 = g->Integral(0,5);
   double r2 = g->Integral(0,1000);
   
   //try with user directives computing more points
   Int_t np = 1000;
   double *x=new double[np];
   double *w=new double[np];
   g->CalcGaussLegendreSamplingPoints(np,x,w,1e-15);
   double r3 = g->IntegralFast(np,x,w,0,5);
   double r4 = g->IntegralFast(np,x,w,0,1000);
   double r5 = g->IntegralFast(np,x,w,0,10000);
   double r6 = g->IntegralFast(np,x,w,0,100000);
   printf("g->Integral(0,5)               = %g\n",r1);
   printf("g->Integral(0,1000)            = %g\n",r2);
   printf("g->IntegralFast(n,x,w,0,5)     = %g\n",r3);
   printf("g->IntegralFast(n,x,w,0,1000)  = %g\n",r4);
   printf("g->IntegralFast(n,x,w,0,10000) = %g\n",r5);
   printf("g->IntegralFast(n,x,w,0,100000)= %g\n",r6);
   delete [] x;
   delete [] w;
}   
</pre>
   <p>This example produces the following results:
<pre>
   g->Integral(0,5)               = 1.25331
   g->Integral(0,1000)            = 1.25319
   g->IntegralFast(n,x,w,0,5)     = 1.25331
   g->IntegralFast(n,x,w,0,1000)  = 1.25331
   g->IntegralFast(n,x,w,0,10000) = 1.25331
   g->IntegralFast(n,x,w,0,100000)= 1.253
</pre>
   <BR><HR>

*/
//End_Html
//---------------------------------------------------------------

   const Double_t kHF = 0.5;
   const Double_t kCST = 5./1000;

   Double_t x[12] = { 0.96028985649753623,  0.79666647741362674,
                      0.52553240991632899,  0.18343464249564980,
                      0.98940093499164993,  0.94457502307323258,
                      0.86563120238783174,  0.75540440835500303,
                      0.61787624440264375,  0.45801677765722739,
                      0.28160355077925891,  0.09501250983763744};

   Double_t w[12] = { 0.10122853629037626,  0.22238103445337447,
                      0.31370664587788729,  0.36268378337836198,
                      0.02715245941175409,  0.06225352393864789,
                      0.09515851168249278,  0.12462897125553387,
                      0.14959598881657673,  0.16915651939500254,
                      0.18260341504492359,  0.18945061045506850};

   Double_t h, aconst, bb, aa, c1, c2, u, s8, s16, f1, f2;
   Double_t xx[1];
   Int_t i;

   InitArgs(xx,params);

   h = 0;
   if (b == a) return h;
   aconst = kCST/TMath::Abs(b-a);
   bb = a;
CASE1:
   aa = bb;
   bb = b;
CASE2:
   c1 = kHF*(bb+aa);
   c2 = kHF*(bb-aa);
   s8 = 0;
   for (i=0;i<4;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = EvalPar(xx,params);
      if (fgAbsValue) f1 = TMath::Abs(f1);
      xx[0] = c1-u;
      f2    = EvalPar(xx,params);
      if (fgAbsValue) f2 = TMath::Abs(f2);
      s8   += w[i]*(f1 + f2);
   }
   s16 = 0;
   for (i=4;i<12;i++) {
      u     = c2*x[i];
      xx[0] = c1+u;
      f1    = EvalPar(xx,params);
      if (fgAbsValue) f1 = TMath::Abs(f1);
      xx[0] = c1-u;
      f2    = EvalPar(xx,params);
      if (fgAbsValue) f2 = TMath::Abs(f2);
      s16  += w[i]*(f1 + f2);
   }
   s16 = c2*s16;
   if (TMath::Abs(s16-c2*s8) <= epsilon*(1. + TMath::Abs(s16))) {
      h += s16;
      if(bb != b) goto CASE1;
   } else {
      bb = c1;
      if(1. + aconst*TMath::Abs(c2) != 1) goto CASE2;
      h = s8;  //this is a crude approximation (cernlib function returned 0 !)
   }
   return h;
}

//______________________________________________________________________________
Double_t TF1::Integral(Double_t, Double_t, Double_t, Double_t, Double_t)
{
   // Return Integral of a 2d function in range [ax,bx],[ay,by]
   //
   Error("Integral","Must be called with a TF2 only");
   return 0;
}

//______________________________________________________________________________
Double_t TF1::Integral(Double_t, Double_t, Double_t, Double_t, Double_t, Double_t, Double_t)
{
   // Return Integral of a 3d function in range [ax,bx],[ay,by],[az,bz]
   //
   Error("Integral","Must be called with a TF3 only");
   return 0;
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
Double_t TF1::IntegralFast(Int_t num, Double_t *x, Double_t *w, Double_t a, Double_t b, Double_t *params)
{
   // Gauss-Legendre integral, see CalcGaussLegendreSamplingPoints
   if (num<=0 || x == 0 || w == 0)
      return 0;

   const Double_t a0 = (b + a)/2;
   const Double_t b0 = (b - a)/2;

   Double_t xx[1];
   InitArgs(xx, params);

   Double_t result = 0.0;
   for (int i=0; i<num; i++)
   {
      xx[0] = a0 + b0*x[i];
      result += w[i] * EvalPar(xx, params);
   }

   return result*b0;
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
//Begin_Html
/*
<img src="gif/integralmultiple.gif">
*/
//End_Html
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
// input parameters
// ================
// n     : Number of dimensions [2,15]
// a,b   : One-dimensional arrays of length >= N . On entry A[i],  and  B[i],
//         contain the lower and upper limits of integration, respectively.
// minpts: Minimum number of function evaluations requested. Must not exceed maxpts. 
//         if minpts < 1 minpts is set to 2^n +2*n*(n+1) +1
// maxpts: Maximum number of function evaluations to be allowed.
//         maxpts >= 2^n +2*n*(n+1) +1
//         if maxpts<minpts, maxpts is set to 10*minpts
// eps   : Specified relative accuracy.
//
// output parameter
// ================
// relerr : Contains, on exit, an estimation of the relative accuracy of the result.
// nfnevl : number of function evaluations performed.
// ifail  :
//     0 Normal exit.  . At least minpts and at most maxpts calls to the function were performed.
//     1 maxpts is too small for the specified accuracy eps. 
//       The result and relerr contain the values obtainable for the 
//       specified value of maxpts.
//     3 n<2 or n>15
//
// Method:
// =======
//
// An integration rule of degree seven is used together with a certain
// strategy of subdivision.
// For a more detailed description of the method see References.
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
//
//=========================================================================
   Double_t ctr[15], wth[15], wthl[15], z[15];

   const Double_t xl2 = 0.358568582800318073;
   const Double_t xl4 = 0.948683298050513796;
   const Double_t xl5 = 0.688247201611685289;
   const Double_t w2  = 980./6561;
   const Double_t w4  = 200./19683;
   const Double_t wp2 = 245./486;
   const Double_t wp4 = 25./729;

   Double_t wn1[14] = {     -0.193872885230909911, -0.555606360818980835,
     -0.876695625666819078, -1.15714067977442459,  -1.39694152314179743,
     -1.59609815576893754,  -1.75461057765584494,  -1.87247878880251983,
     -1.94970278920896201,  -1.98628257887517146,  -1.98221815780114818,
     -1.93750952598689219,  -1.85215668343240347,  -1.72615963013768225};

   Double_t wn3[14] = {     0.0518213686937966768,  0.0314992633236803330,
     0.0111771579535639891,-0.00914494741655235473,-0.0294670527866686986,
    -0.0497891581567850424,-0.0701112635269013768, -0.0904333688970177241,
    -0.110755474267134071, -0.131077579637250419,  -0.151399685007366752,
    -0.171721790377483099, -0.192043895747599447,  -0.212366001117715794};

   Double_t wn5[14] = {         0.871183254585174982e-01,  0.435591627292587508e-01,
     0.217795813646293754e-01,  0.108897906823146873e-01,  0.544489534115734364e-02,
     0.272244767057867193e-02,  0.136122383528933596e-02,  0.680611917644667955e-03,
     0.340305958822333977e-03,  0.170152979411166995e-03,  0.850764897055834977e-04,
     0.425382448527917472e-04,  0.212691224263958736e-04,  0.106345612131979372e-04};

   Double_t wpn1[14] = {   -1.33196159122085045, -2.29218106995884763,
     -3.11522633744855959, -3.80109739368998611, -4.34979423868312742,
     -4.76131687242798352, -5.03566529492455417, -5.17283950617283939,
     -5.17283950617283939, -5.03566529492455417, -4.76131687242798352,
     -4.34979423868312742, -3.80109739368998611, -3.11522633744855959};

   Double_t wpn3[14] = {     0.0445816186556927292, -0.0240054869684499309,
    -0.0925925925925925875, -0.161179698216735251,  -0.229766803840877915,
    -0.298353909465020564,  -0.366941015089163228,  -0.435528120713305891,
    -0.504115226337448555,  -0.572702331961591218,  -0.641289437585733882,
    -0.709876543209876532,  -0.778463648834019195,  -0.847050754458161859};

   Double_t result = 0;
   Double_t abserr = 0;
   ifail  = 3;
   nfnevl = 0;
   relerr = 0;
   if (n < 2 || n > 15) return 0;

   Double_t twondm = TMath::Power(2,n);
   Int_t ifncls = 0;
   Bool_t ldv   = kFALSE;
   Int_t irgnst = 2*n+3;
   Int_t irlcls = Int_t(twondm) +2*n*(n+1)+1;
   Int_t isbrgn = irgnst;
   Int_t isbrgs = irgnst;

   if (minpts < 1)      minpts = irlcls;
   if (maxpts < minpts) maxpts = 10*minpts;

// The original agorithm expected a working space array WK of length IWK
// with IWK Length ( >= (2N + 3) * (1 + MAXPTS/(2**N + 2N(N + 1) + 1))/2).
// Here, this array is allocated dynamically

   Int_t iwk = irgnst*(1 +maxpts/irlcls)/2;
   Double_t *wk = new Double_t[iwk+10];
   Int_t j;
   for (j=0;j<n;j++) {
      ctr[j] = (b[j] + a[j])*0.5;
      wth[j] = (b[j] - a[j])*0.5;
   }

   Double_t rgnvol, sum1, sum2, sum3, sum4, sum5, difmax, f2, f3, dif, aresult;
   Double_t rgncmp=0, rgnval, rgnerr;
   Int_t j1, k, l, m, idvaxn=0, idvax0=0, isbtmp, isbtpp;

   InitArgs(z,fParams);

L20:
   rgnvol = twondm;
   for (j=0;j<n;j++) {
      rgnvol *= wth[j];
      z[j]    = ctr[j];
   }
   sum1 = EvalPar(z,fParams); //evaluate function

   difmax = 0;
   sum2   = 0;
   sum3   = 0;
   for (j=0;j<n;j++) {
      z[j]    = ctr[j] - xl2*wth[j];
      if (fgAbsValue) f2 = TMath::Abs(EvalPar(z,fParams));
      else            f2 = EvalPar(z,fParams);
      z[j]    = ctr[j] + xl2*wth[j];
      if (fgAbsValue) f2 += TMath::Abs(EvalPar(z,fParams));
      else            f2 += EvalPar(z,fParams);
      wthl[j] = xl4*wth[j];
      z[j]    = ctr[j] - wthl[j];
      if (fgAbsValue) f3 = TMath::Abs(EvalPar(z,fParams));
      else            f3 = EvalPar(z,fParams);
      z[j]    = ctr[j] + wthl[j];
      if (fgAbsValue) f3 += TMath::Abs(EvalPar(z,fParams));
      else            f3 += EvalPar(z,fParams);
      sum2   += f2;
      sum3   += f3;
      dif     = TMath::Abs(7*f2-f3-12*sum1);
      if (dif >= difmax) {
         difmax=dif;
         idvaxn=j+1;
      }
      z[j]    = ctr[j];
   }

   sum4 = 0;
   for (j=1;j<n;j++) {
      j1 = j-1;
      for (k=j;k<n;k++) {
         for (l=0;l<2;l++) {
            wthl[j1] = -wthl[j1];
            z[j1]    = ctr[j1] + wthl[j1];
            for (m=0;m<2;m++) {
               wthl[k] = -wthl[k];
               z[k]    = ctr[k] + wthl[k];
               if (fgAbsValue) sum4 += TMath::Abs(EvalPar(z,fParams));
               else            sum4 += EvalPar(z,fParams);
            }
         }
         z[k] = ctr[k];
      }
      z[j1] = ctr[j1];
   }

   sum5 = 0;
   for (j=0;j<n;j++) {
      wthl[j] = -xl5*wth[j];
      z[j] = ctr[j] + wthl[j];
   }
L90:
   if (fgAbsValue) sum5 += TMath::Abs(EvalPar(z,fParams));
   else            sum5 += EvalPar(z,fParams);
   for (j=0;j<n;j++) {
      wthl[j] = -wthl[j];
      z[j] = ctr[j] + wthl[j];
      if (wthl[j] > 0) goto L90;
   }

   rgncmp  = rgnvol*(wpn1[n-2]*sum1+wp2*sum2+wpn3[n-2]*sum3+wp4*sum4);
   rgnval  = wn1[n-2]*sum1+w2*sum2+wn3[n-2]*sum3+w4*sum4+wn5[n-2]*sum5;
   rgnval *= rgnvol;
   rgnerr  = TMath::Abs(rgnval-rgncmp);
   result += rgnval;
   abserr += rgnerr;
   ifncls += irlcls;
   aresult = TMath::Abs(result);
   //if (result > 0 && aresult< 1e-100) {
   //   delete [] wk;
   //   ifail = 0;  //function is probably symmetric ==> integral is null: not an error
   //   return result;
   //}

   if (ldv) {
L110:
      isbtmp = 2*isbrgn;
      if (isbtmp > isbrgs) goto L160;
      if (isbtmp < isbrgs) {
         isbtpp = isbtmp + irgnst;
         if (wk[isbtmp-1] < wk[isbtpp-1]) isbtmp = isbtpp;
      }
      if (rgnerr >= wk[isbtmp-1]) goto L160;
      for (k=0;k<irgnst;k++) {
         wk[isbrgn-k-1] = wk[isbtmp-k-1];
      }
      isbrgn = isbtmp;
      goto L110;
   }
L140:
   isbtmp = (isbrgn/(2*irgnst))*irgnst;
   if (isbtmp >= irgnst && rgnerr > wk[isbtmp-1]) {
      for (k=0;k<irgnst;k++) {
         wk[isbrgn-k-1] = wk[isbtmp-k-1];
      }
      isbrgn = isbtmp;
      goto L140;
   }

L160:
   wk[isbrgn-1] = rgnerr;
   wk[isbrgn-2] = rgnval;
   wk[isbrgn-3] = Double_t(idvaxn);
   for (j=0;j<n;j++) {
      isbtmp = isbrgn-2*j-4;
      wk[isbtmp]   = ctr[j];
      wk[isbtmp-1] = wth[j];
   }
   if (ldv) {
      ldv = kFALSE;
      ctr[idvax0-1] += 2*wth[idvax0-1];
      isbrgs += irgnst;
      isbrgn  = isbrgs;
      goto L20;
   }
   relerr = abserr/aresult;
   if (relerr < 1e-1 && aresult < 1e-20) ifail = 0;
   if (relerr < 1e-3 && aresult < 1e-10) ifail = 0;
   if (relerr < 1e-5 && aresult < 1e-5)  ifail = 0;
   if (isbrgs+irgnst > iwk) ifail = 2;
   if (ifncls+2*irlcls > maxpts) {
      if (sum1==0 && sum2==0 && sum3==0 && sum4==0 && sum5==0){
         ifail = 0;
         result = 0;
      }
      else
         ifail = 1;
   }
   if (relerr < eps && ifncls >= minpts) ifail = 0;
   if (ifail == 3) {
      ldv = kTRUE;
      isbrgn  = irgnst;
      abserr -= wk[isbrgn-1];
      result -= wk[isbrgn-2];
      idvax0  = Int_t(wk[isbrgn-3]);
      for (j=0;j<n;j++) {
         isbtmp = isbrgn-2*j-4;
         ctr[j] = wk[isbtmp];
         wth[j] = wk[isbtmp-1];
      }
      wth[idvax0-1]  = 0.5*wth[idvax0-1];
      ctr[idvax0-1] -= wth[idvax0-1];
      goto L20;
   }
   delete [] wk;
   nfnevl = ifncls;       //number of function evaluations performed.
   return result;         //an approximate value of the integral
}

//______________________________________________________________________________
Bool_t TF1::IsInside(const Double_t *x) const
{
   // Return kTRUE is the point is inside the function range

   if (x[0] < fXmin || x[0] > fXmax) return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
void TF1::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this function with its current attributes*-*-*-*-*
//*-*                  ===============================================

   Int_t i;
   Double_t xv[1];

   fgCurrent = this;
   
   TString opt = option;
   opt.ToLower();
   Double_t xmin=fXmin, xmax=fXmax, pmin=fXmin, pmax=fXmax;
   if (gPad) {
      pmin = gPad->PadtoX(gPad->GetUxmin());
      pmax = gPad->PadtoX(gPad->GetUxmax());
   }
   if (opt.Contains("same")) {
      if (xmax < pmin) return;  // Otto: completely outside
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
      strcpy(ctemp,semicol+1);
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
//      if logx, we must bin in logx and not in x !!!
//      otherwise if several decades, one gets crazy results
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
   //restore axis titles
   fHistogram->GetXaxis()->SetTitle(xtitle.Data());
   fHistogram->GetYaxis()->SetTitle(ytitle.Data());
   
   InitArgs(xv,fParams);
   for (i=1;i<=fNpx;i++) {
      xv[0] = fHistogram->GetBinCenter(i);
      fHistogram->SetBinContent(i,EvalPar(xv,fParams));
   }

//*-*- Copy Function attributes to histogram attributes
   Double_t minimum   = fHistogram->GetMinimumStored();
   Double_t maximum   = fHistogram->GetMaximumStored();
   if (minimum <= 0 && gPad && gPad->GetLogy()) minimum = -1111; //this can happen when switching from lin to log scale
   if (minimum == -1111) { //this can happen after unzooming
      if (fHistogram->TestBit(TH1::kIsZoomed)) {
         minimum = fHistogram->GetYaxis()->GetXmin();
      } else {
         minimum = fMinimum;
         if (minimum == -1111) {
            Double_t hmin = fHistogram->GetMinimum();
            if (hmin > 0) {
               Double_t hmax = fHistogram->GetMaximum();
               hmin -= 0.05*(hmax-hmin);
               if (hmin < 0) hmin = 0;
               if (hmin <= 0 && gPad && gPad->GetLogy()) hmin = 0.001*hmax;
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

//*-*-  Draw the histogram
   if (!gPad) return;
   if (opt.Length() == 0)  fHistogram->Paint("lf");
   else if (opt == "same") fHistogram->Paint("lfsame");
   else                    fHistogram->Paint(option);

}

//______________________________________________________________________________
void TF1::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*Dump this function with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  ==================================
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
   fNsave = fNpx+3;
   if (fNsave <= 3) {fNsave=0; return;}
   fSave  = new Double_t[fNsave];
   Int_t i;
   Double_t dx = (xmax-xmin)/fNpx;
   if (dx <= 0) {
      dx = (fXmax-fXmin)/fNpx;
      fNsave--;
      xmin = fXmin +0.5*dx;
      xmax = fXmax -0.5*dx;
   }
   Double_t xv[1];
   InitArgs(xv,fParams);
   for (i=0;i<=fNpx;i++) {
      xv[0]    = xmin + dx*i;
      fSave[i] = EvalPar(xv,fParams);
   }
   fSave[fNpx+1] = xmin;
   fSave[fNpx+2] = xmax;
}

//______________________________________________________________________________
void TF1::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

   Int_t i;
   char quote = '"';
   out<<"   "<<endl;
   if (!fMethodCall) {
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
   // static function setting the current function.
   // the current function may be accessed in static C-like functions
   // when fitting or painting a function.
   
   fgCurrent = f1;
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
// the minimum number of points is 4, the maximum is 100000 for 1-d and 10000 for 2-d/3-d functions

   if (npx < 4) {
      Warning("SetNpx","Number of points must be >4 && < 100000, fNpx set to 4");
      fNpx = 4;
   } else if(npx > 100000) {
      Warning("SetNpx","Number of points must be >4 && < 100000, fNpx set to 100000");
      fNpx = 100000;
   } else {
      fNpx = npx;
   }
   Update();
}

//______________________________________________________________________________
void TF1::SetParError(Int_t ipar, Double_t error)
{
// set error for parameter number ipar

   if (ipar < 0 || ipar > fNpar-1) return;
   fParErrors[ipar] = error;
}

//______________________________________________________________________________
void TF1::SetParErrors(const Double_t *errors)
{
// set errors for all active parameters
// when calling this function, the array errors must have at least fNpar values

   if (!errors) return;
   for (Int_t i=0;i<fNpar;i++) fParErrors[i] = errors[i];
}

//______________________________________________________________________________
void TF1::SetParLimits(Int_t ipar, Double_t parmin, Double_t parmax)
{
//*-*-*-*-*-*Set limits for parameter ipar*-*-*-*
//*-*        =============================
//     The specified limits will be used in a fit operation
//     when the option "B" is specified (Bounds).
//  To fix a parameter, use TF1::FixParameter

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
//*-*-*-*-*-*Initialize the upper and lower bounds to draw the function*-*-*-*
//*-*        ==========================================================
//     The function range is also used in an histogram fit operation
//     when the option "R" is specified.

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
   char *semicol = (char*)strstr(title,";");
   if (semicol) {
      Int_t nxt = strlen(semicol);
      char *ctemp = new char[nxt];
      strcpy(ctemp,semicol+1);
      semicol = (char*)strstr(ctemp,";");
      if (semicol) {
         *semicol = 0;
         fHistogram->GetYaxis()->SetTitle(semicol+1);
      }
      fHistogram->GetXaxis()->SetTitle(ctemp);
      delete [] ctemp;
   }
   if (gPad) gPad->Modified();
}

//_______________________________________________________________________
void TF1::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 4) {
         TF1::Class()->ReadBuffer(b, this, v, R__s, R__c);
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

      TF1::Class()->WriteBuffer(b,this);

      if (saved) {delete [] fSave; fSave = 0; fNsave = 0;}
   }
}

//_______________________________________________________________________
void TF1::Update()
{
// called by functions such as SetRange, SetNpx, SetParameters
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

//_______________________________________________________________________
void TF1::RejectPoint(Bool_t reject)
{
// static function to set the global flag to reject points
// the fgRejectPoint global flag is tested by all fit functions
// if TRUE the point is not included in the fit.
// This flag can be set by a user in a fitting function.
// The fgRejectPoint flag is reset by the TH1 and TGraph fitting functions.

   fgRejectPoint = reject;
}

//_______________________________________________________________________
Bool_t TF1::RejectedPoint()
{
// see TF1::RejectPoint above
   return fgRejectPoint;
}

//______________________________________________________________________________
Double_t TF1_ExpValHelperx(Double_t *x, Double_t *par) {
   return x[0]*gHelper->EvalPar(x,par);
}

//______________________________________________________________________________
Double_t TF1_ExpValHelper(Double_t *x, Double_t *par) {
   Int_t npar    = gHelper->GetNpar();
   Double_t xbar = par[npar];
   Double_t n    = par[npar+1];
   return TMath::Power(x[0]-xbar,n)*gHelper->EvalPar(x,par);
}


//______________________________________________________________________________
Double_t TF1::Moment(Double_t n, Double_t a, Double_t b, const Double_t *params, Double_t epsilon)
{
// Return nth moment of function between a and b
//
// See TF1::Integral() for parameter definitions
//   Author: Gene Van Buren <gene@bnl.gov>

   fgAbsValue = kTRUE;
   Double_t norm = Integral(a,b,params,epsilon);
   if (norm == 0) {
      fgAbsValue = kFALSE;
      Error("Moment", "Integral zero over range");
      return 0;
   }

   gHelper = this;
   //TF1 fnc("TF1_ExpValHelper",Form("%s*pow(x,%f)",GetName(),n));
   TF1 fnc("TF1_ExpValHelper",TF1_ExpValHelper,fXmin,fXmax,fNpar+2);
   for (Int_t i=0;i<fNpar;i++) {
      if(params) fnc.SetParameter(i,params[i]);
      else       fnc.SetParameter(i,fParams[i]);
   }
   fnc.SetParameter(fNpar,0);
   fnc.SetParameter(fNpar+1,n);
   Double_t res = fnc.Integral(a,b,params,epsilon)/norm;
   fgAbsValue = kFALSE;
   return res;
}

//______________________________________________________________________________
Double_t TF1::CentralMoment(Double_t n, Double_t a, Double_t b, const Double_t *params, Double_t epsilon)
{
// Return nth central moment of function between a and b
//
// See TF1::Integral() for parameter definitions
//   Author: Gene Van Buren <gene@bnl.gov>

   fgAbsValue = kTRUE;
   Double_t norm = Integral(a,b,params,epsilon);
   if (norm == 0) {
      fgAbsValue = kFALSE;
      Error("CentralMoment", "Integral zero over range");
      return 0;
   }

   gHelper = this;
   //TF1 fncx("TF1_ExpValHelperx",Form("%s*x",GetName()));
   TF1 fncx("TF1_ExpValHelperx",TF1_ExpValHelperx,fXmin,fXmax,fNpar);
   Int_t i;
   for (i=0;i<fNpar;i++) {
      if(params) fncx.SetParameter(i,params[i]);
      else       fncx.SetParameter(i,fParams[i]);
   }
   Double_t xbar = fncx.Integral(a,b,params,epsilon)/norm;

   //TF1 fnc("TF1_ExpValHelper",Form("%s*pow(x-%f,%f)",GetName(),xbar,n));
   TF1 fnc("TF1_ExpValHelper",TF1_ExpValHelper,fXmin,fXmax,fNpar+2);
   for (i=0;i<fNpar;i++) {
      if(params) fnc.SetParameter(i,params[i]);
      else       fnc.SetParameter(i,fParams[i]);
   }
   fnc.SetParameter(fNpar,0);
   fnc.SetParameter(fNpar+1,n);
   fnc.SetParameter(fNpar,xbar);
   fnc.SetParameter(fNpar+1,n);
   Double_t res = fnc.Integral(a,b,params,epsilon)/norm;
   fgAbsValue = kFALSE;
   return res;
}

//--------------------------------------------------------------------
// some useful static utility functions to compute sampling points for IntegralFast
//--------------------------------------------------------------------
//______________________________________________________________________________
#ifdef INTHEFUTURE
void TF1::CalcGaussLegendreSamplingPoints(TGraph *g, Double_t eps)
{
   //type safe interface (static method)
   // The number of sampling points are taken from the TGraph
   if (!g) return;
   CalcGaussLegendreSamplingPoints(g->GetN(), g->GetX(), g->GetY(), eps);
}

//______________________________________________________________________________
TGraph *TF1::CalcGaussLegendreSamplingPoints(Int_t num, Double_t eps)
{
   //type safe interface (static method)
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
   //
   if (num<=0 || eps<=0)
      return;

   // The roots of symmetric is the interval, so we only have to find half of them
   const UInt_t m = (num+1)/2;

   Double_t z, pp, p1,p2, p3;

   // Loop over the disired roots
   for (UInt_t i=0; i<m; i++) {
      z = TMath::Cos(TMath::Pi()*(i+0.75)/(num+0.5));

      // Starting with the above approximation to the i-th root, we enter
      // the main loop of refinement by Newton's method
      do {
         p1=1.0;
         p2=0.0;

         // Loop up the recurrence relation to get the Legendre
         // polynomial evaluated at z
         for (int j=0; j<num; j++)
         {
            p3 = p2;
            p2 = p1;
            p1 = ((2.0*j+1.0)*z*p2-j*p3)/(j+1.0);
         }
         // p1 is now the desired Legendre polynomial. We next compute pp, its
         // derivative, by a standard relation involving also p2, the polynomial
         // of one lower order
         pp = num*(z*p1-p2)/(z*z-1.0);
         // Newton's method
         z -= p1/pp;

      } while (TMath::Abs(p1/pp) > eps);

      // Put root and its symmetric counterpart
      x[i]       = -z;
      x[num-i-1] =  z;

      // Compute the weight and put its symmetric counterpart
      w[i]       = 2.0/((1.0-z*z)*pp*pp);
      w[num-i-1] = w[i];
   }
}
