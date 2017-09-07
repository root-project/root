// @(#)root/hist:$Id$
// Author: Marian Ivanov, 2005

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "v5/TFormulaPrimitive.h"

#include "TMath.h"
#include "TNamed.h"
#include "TObjArray.h"
#include "TVirtualMutex.h"

#include <math.h>

#ifdef WIN32
#pragma optimize("",off)
#endif

static TVirtualMutex* gTFormulaPrimativeListMutex = 0;


ClassImp(ROOT::v5::TFormulaPrimitive);

namespace ROOT  {
   namespace v5 {

      void TMath_GenerInterface();

/** \class TFormulaPrimitive  TFormulaPrimitive.h "inc/v5/TFormulaPrimitive.h"
     \ingroup Hist
The Formula Primitive class

Helper class for TFormula to speed up TFormula evaluation
TFormula can use all functions registered in the list of TFormulaPrimitives
User can add new function to the list of primitives
if FormulaPrimitive with given name is already defined new primitive is ignored

Example:

~~~ {.cpp}
     TFormulaPrimitive::AddFormula(new TFormulaPrimitive("Pow2","Pow2",TFastFun::Pow2));
     TF1 f1("f1","Pow2(x)");
~~~

  - TFormulaPrimitive is used to get direct acces to the function pointers
  - GenFunc     -  pointers  to the static function
  - TFunc       -  pointers  to the data member functions

The following sufixes are currently used, to describe function arguments:

  - G     - generic layout - pointer to double (arguments), pointer to double (parameters)
  - 10    - double
  - 110   - double, double
  - 1110  - double, double, double
*/

//______________________________________________________________________________
// TFormula primitive
//
TObjArray * TFormulaPrimitive::fgListOfFunction = 0;
#ifdef R__COMPLETE_MEM_TERMINATION
namespace {
   class TFormulaPrimitiveCleanup {
      TObjArray **fListOfFunctions;
   public:
      TFormulaPrimitiveCleanup(TObjArray **functions) : fListOfFunctions(functions) {}
      ~TFormulaPrimitiveCleanup() {
         delete *fListOfFunctions;
      }
   };

}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TFormulaPrimitive::TFormulaPrimitive() : TNamed(),
                                         fFuncG(0),
                                         fType(0),fNArguments(0),fNParameters(0),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     GenFunc0 fpointer) : TNamed(name,formula),
                                                          fFunc0(fpointer),
                                                          fType(0),fNArguments(0),fNParameters(0),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     GenFunc10 fpointer) : TNamed(name,formula),
                                                           fFunc10(fpointer),
                                                           fType(10),fNArguments(1),fNParameters(0),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     GenFunc110 fpointer) : TNamed(name,formula),
                                                            fFunc110(fpointer),
                                                            fType(110),fNArguments(2),fNParameters(0),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     GenFunc1110 fpointer) : TNamed(name,formula),
                                                             fFunc1110(fpointer),
                                                             fType(1110),fNArguments(3),fNParameters(0),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     GenFuncG fpointer,Int_t npar) : TNamed(name,formula),
                                                                     fFuncG(fpointer),
                                                                     fType(-1),fNArguments(2),fNParameters(npar),fIsStatic(kTRUE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     TFuncG fpointer) : TNamed(name,formula),
                                                        fTFuncG(fpointer),
                                                        fType(0),fNArguments(0),fNParameters(0),fIsStatic(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     TFunc0 fpointer) : TNamed(name,formula),
                                                        fTFunc0(fpointer),
                                                        fType(0),fNArguments(0),fNParameters(0),fIsStatic(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     TFunc10 fpointer) : TNamed(name,formula),
                                                         fTFunc10(fpointer),
                                                         fType(-10),fNArguments(1),fNParameters(0),fIsStatic(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     TFunc110 fpointer) : TNamed(name,formula),
                                                          fTFunc110(fpointer),
                                                          fType(-110),fNArguments(2),fNParameters(0),fIsStatic(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormulaPrimitive::TFormulaPrimitive(const char *name,const char *formula,
                                     TFunc1110 fpointer) :TNamed(name,formula),
                                                          fTFunc1110(fpointer),
                                                          fType(-1110),fNArguments(3),fNParameters(0),fIsStatic(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Add formula to the list of primitive formulas.
/// If primitive formula already defined do nothing.

Int_t TFormulaPrimitive::AddFormula(TFormulaPrimitive * formula)
{
   R__LOCKGUARD2(gTFormulaPrimativeListMutex);
   if (fgListOfFunction == 0) BuildBasicFormulas();
   if (FindFormula(formula->GetName(),formula->fNArguments)){
      delete formula;
      return 0;
   }
   fgListOfFunction->AddLast(formula);
   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Eval primitive function at point x.

Double_t TFormulaPrimitive::Eval(Double_t* x)
{
   if (fIsStatic == kFALSE) return 0;

   if (fType==0) return  fFunc0();
   if (fType==10) {
      return fFunc10(x[0]);
   }
   if (fType==110) {
      return fFunc110(x[0],x[1]);
   }
   if (fType==1110) {
      return fFunc1110(x[0],x[1],x[2]);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Eval member function of object o at point x.

Double_t  TFormulaPrimitive::Eval(TObject *o, Double_t *x)
{
   if (fIsStatic == kTRUE) return 0;
   if (fType== 0)    return (*o.*fTFunc0)();
   if (fType==-10)   return (*o.*fTFunc10)(*x);
   if (fType==-110)  return (*o.*fTFunc110)(x[0],x[1]);
   if (fType==-1110) return (*o.*fTFunc1110)(x[0],x[1],x[2]);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Eval primitive parametric function.

Double_t TFormulaPrimitive::Eval(Double_t *x, Double_t *param)
{
   return fFuncG(x,param);
}

#define RTFastFun__POLY(var)                                          \
{                                                                     \
   Double_t res= param[var-1]+param[var]*x[0];                        \
   for (Int_t j=var-1 ;j>0;j--) res      = param[j-1]+x[0]*res;       \
   return res;                                                        \
}

namespace TFastFun {
   //
   // Namespace with basic primitive functions registered by TFormulaPrimitive
   // all function registered by TFormulaPrimitive can be used in TFormula
   //
   Double_t Pow2(Double_t x){return x*x;}
   Double_t Pow3(Double_t x){return x*x*x;}
   Double_t Pow4(Double_t x){return x*x*x*x;}
   Double_t Pow5(Double_t x){return x*x*x*x*x;}
   inline   Double_t FPoln(Double_t *x, Double_t *param, Int_t npar);
   Double_t FPol0(Double_t * /*x*/, Double_t *param){ return param[0];}
   Double_t FPol1(Double_t *x, Double_t *param){ return param[0]+param[1]*x[0];}
   Double_t FPol2(Double_t *x, Double_t *param){ return param[0]+x[0]*(param[1]+param[2]*x[0]);}
   Double_t FPol3(Double_t *x, Double_t *param){ return param[0]+x[0]*(param[1]+x[0]*(param[2]+param[3]*x[0]));}
   Double_t FPol4(Double_t *x, Double_t *param){ RTFastFun__POLY(4)}
   Double_t FPol5(Double_t *x, Double_t *param){ RTFastFun__POLY(5)}
   Double_t FPol6(Double_t *x, Double_t *param){ RTFastFun__POLY(6)}
   Double_t FPol7(Double_t *x, Double_t *param){ RTFastFun__POLY(7)}
   Double_t FPol8(Double_t *x, Double_t *param){ RTFastFun__POLY(8)}
   Double_t FPol9(Double_t *x, Double_t *param){ RTFastFun__POLY(9)}
   Double_t FPol10(Double_t *x, Double_t *param){ RTFastFun__POLY(10)}
   //
   //
   Double_t PlusXY(Double_t x,Double_t y){return x+y;}
   Double_t MinusXY(Double_t x,Double_t y){return x-y;}
   Double_t MultXY(Double_t x,Double_t y){return x*y;}
   Double_t DivXY(Double_t x, Double_t y){return TMath::Abs(y)>0 ? x/y:0;}
   Double_t XpYpZ(Double_t x, Double_t y, Double_t z){ return x+y+z;}
   Double_t XxYxZ(Double_t x, Double_t y, Double_t z){ return x*y*z;}
   Double_t XxYpZ(Double_t x, Double_t y, Double_t z){ return x*(y+z);}
   Double_t XpYxZ(Double_t x, Double_t y, Double_t z){ return x+(y*z);}
   Double_t Gaus(Double_t x, Double_t mean, Double_t sigma);
   Double_t Gausn(Double_t x, Double_t mean, Double_t sigma);
   Double_t Landau(Double_t x, Double_t mean, Double_t sigma){return TMath::Landau(x,mean,sigma,kFALSE);}
   Double_t Landaun(Double_t x, Double_t mean, Double_t sigma){return TMath::Landau(x,mean,sigma,kTRUE);}
   Double_t Sqrt(Double_t x) {return x>0?sqrt(x):0;}
   //
   Double_t Sign(Double_t x){return (x<0)? -1:1;}
   Double_t Nint(Double_t x){return TMath::Nint(x);}
   Double_t Abs(Double_t x){return TMath::Abs(x);}
   //logical
   Double_t XandY(Double_t x, Double_t y){ return (x*y>0.1);}
   Double_t XorY(Double_t x, Double_t y) { return (x+y>0.1);}
   Double_t XgY(Double_t x, Double_t y) {return (x>y);}
   Double_t XgeY(Double_t x, Double_t y) {return (x>=y);}
   Double_t XlY(Double_t x, Double_t y) {return (x<y);}
   Double_t XleY(Double_t x, Double_t y) {return (x<=y);}
   Double_t XeY(Double_t x,Double_t y) {return (x==y);}
   Double_t XneY(Double_t x,Double_t y) {return (x!=y);}
   Double_t XNot(Double_t x){ return (x<0.1);}
};

////////////////////////////////////////////////////////////////////////////////
/// Find the formula in the list of formulas.

TFormulaPrimitive* TFormulaPrimitive::FindFormula(const char* name)
{
   R__LOCKGUARD2(gTFormulaPrimativeListMutex);
   if (!fgListOfFunction) {
      BuildBasicFormulas();
   }
   Int_t nobjects = fgListOfFunction->GetEntries();
   for (Int_t i = 0; i < nobjects; ++i) {
      TFormulaPrimitive *formula = (TFormulaPrimitive*)fgListOfFunction->At(i);
      if (formula && 0==strcmp(name, formula->GetName())) return formula;
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the formula in the list of formulas.

TFormulaPrimitive* TFormulaPrimitive::FindFormula(const char* name, UInt_t nargs)
{
   R__LOCKGUARD2(gTFormulaPrimativeListMutex);
   if (!fgListOfFunction) {
      BuildBasicFormulas();
   }
   Int_t nobjects = fgListOfFunction->GetEntries();
   for (Int_t i = 0; i < nobjects; ++i) {
      TFormulaPrimitive *prim = (TFormulaPrimitive*)fgListOfFunction->At(i);
      if (prim) {
         bool match = ( ((UInt_t)prim->fNArguments) == nargs );
         if (match && 0==strcmp(name, prim->GetName())) return prim;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the formula in the list of formulas.

TFormulaPrimitive* TFormulaPrimitive::FindFormula(const char* name, const char *args)
{
   // let's count the argument(s)
   if (args) {
      Int_t nargs = 0;
      if (args[0]!=')') {
         nargs = 1;
         int nest = 0;
         for(UInt_t c = 0; c < strlen(args); ++c ) {
            switch (args[c]) {
               case '(': ++nest; break;
               case ')': --nest; break;
               case '<': ++nest; break;
               case '>': --nest; break;
               case ',': nargs += (nest==0); break;
            }
         }
      }
      return FindFormula(name,nargs);
   } else {
      return FindFormula(name);
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// FPoln.

Double_t TFastFun::FPoln(Double_t *x, Double_t *param, Int_t npar)
{
   Double_t res = 0; Double_t temp=1;
   for (Int_t j=npar ;j>=0;j--) {
      res  += temp*param[j];
      temp *= *x;
   }
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Gauss.

Double_t TFastFun::Gaus(Double_t x, Double_t mean, Double_t sigma)
{
   if (sigma == 0) return 1.e30;
   Double_t arg = (x-mean)/sigma;
   return TMath::Exp(-0.5*arg*arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Normalize gauss.

Double_t TFastFun::Gausn(Double_t x, Double_t mean, Double_t sigma)
{
   if (sigma == 0)  return 0;
   Double_t arg = (x-mean)/sigma;
   return TMath::Exp(-0.5*arg*arg)/(2.50662827463100024*sigma);  //sqrt(2*Pi)=2.50662827463100024
}

////////////////////////////////////////////////////////////////////////////////
/// Built-in functions.

Int_t TFormulaPrimitive::BuildBasicFormulas()
{
   R__LOCKGUARD2(gTFormulaPrimativeListMutex);
   if (fgListOfFunction==0) {
      fgListOfFunction = new TObjArray(1000);
      fgListOfFunction->SetOwner(kTRUE);
   }
#ifdef R__COMPLETE_MEM_TERMINATION
   static TFormulaPrimitiveCleanup gCleanup(&fgListOfFunction);
#endif

   //
   // logical
   //
   AddFormula(new TFormulaPrimitive("XandY","XandY",TFastFun::XandY));
   AddFormula(new TFormulaPrimitive("XorY","XorY",TFastFun::XorY));
   AddFormula(new TFormulaPrimitive("XNot","XNot",TFastFun::XNot));
   AddFormula(new TFormulaPrimitive("XlY","XlY",TFastFun::XlY));
   AddFormula(new TFormulaPrimitive("XleY","XleY",TFastFun::XleY));
   AddFormula(new TFormulaPrimitive("XgY","XgY",TFastFun::XgY));
   AddFormula(new TFormulaPrimitive("XgeY","XgeY",TFastFun::XgeY));
   AddFormula(new TFormulaPrimitive("XeY","XeY",TFastFun::XeY));
   AddFormula(new TFormulaPrimitive("XneY","XneY",TFastFun::XneY));
   // addition  + multiplication
   AddFormula(new TFormulaPrimitive("PlusXY","PlusXY",TFastFun::PlusXY));
   AddFormula(new TFormulaPrimitive("MinusXY","MinusXY",TFastFun::MinusXY));
   AddFormula(new TFormulaPrimitive("MultXY","MultXY",TFastFun::MultXY));
   AddFormula(new TFormulaPrimitive("DivXY","DivXY",TFastFun::DivXY));
   AddFormula(new TFormulaPrimitive("XpYpZ","XpYpZ",TFastFun::XpYpZ));
   AddFormula(new TFormulaPrimitive("XxYxZ","XxYxZ",TFastFun::XxYxZ));
   AddFormula(new TFormulaPrimitive("XxYpZ","XxYpZ",TFastFun::XxYpZ));
   AddFormula(new TFormulaPrimitive("XpYxZ","XpYxZ",TFastFun::XpYxZ));
   //
   //
   AddFormula(new TFormulaPrimitive("Gaus","Gaus",TFastFun::Gaus));
   AddFormula(new TFormulaPrimitive("Gausn","Gausn",TFastFun::Gausn));
   AddFormula(new TFormulaPrimitive("Landau","Landau",TFastFun::Landau));
   AddFormula(new TFormulaPrimitive("Landaun","Landaun",TFastFun::Landaun));
   //
   //
   // polynoms
   //
   //
   AddFormula(new TFormulaPrimitive("Pol0","Pol0",(GenFuncG)TFastFun::FPol0,1));
   AddFormula(new TFormulaPrimitive("Pol1","Pol1",(GenFuncG)TFastFun::FPol1,2));
   AddFormula(new TFormulaPrimitive("Pol2","Pol2",(GenFuncG)TFastFun::FPol2,3));
   AddFormula(new TFormulaPrimitive("Pol3","Pol3",(GenFuncG)TFastFun::FPol3,4));
   AddFormula(new TFormulaPrimitive("Pol4","Pol4",(GenFuncG)TFastFun::FPol4,5));
   AddFormula(new TFormulaPrimitive("Pol5","Pol5",(GenFuncG)TFastFun::FPol5,6));
   AddFormula(new TFormulaPrimitive("Pol6","Pol6",(GenFuncG)TFastFun::FPol6,7));
   AddFormula(new TFormulaPrimitive("Pol7","Pol7",(GenFuncG)TFastFun::FPol7,8));
   AddFormula(new TFormulaPrimitive("Pol8","Pol8",(GenFuncG)TFastFun::FPol8,9));
   AddFormula(new TFormulaPrimitive("Pol9","Pol9",(GenFuncG)TFastFun::FPol9,10));
   AddFormula(new TFormulaPrimitive("Pol10","Pol10",(GenFuncG)TFastFun::FPol10,11));
   //
   // pows
   AddFormula(new TFormulaPrimitive("Pow2","Pow2",TFastFun::Pow2));
   AddFormula(new TFormulaPrimitive("Pow3","Pow3",TFastFun::Pow3));
   AddFormula(new TFormulaPrimitive("Pow4","Pow4",TFastFun::Pow4));
   AddFormula(new TFormulaPrimitive("Pow5","Pow5",TFastFun::Pow5));
   //
   //
   AddFormula(new TFormulaPrimitive("TMath::Cos","TMath::Cos",cos));              // 10
   AddFormula(new TFormulaPrimitive("cos","cos",cos));                            // 10
   AddFormula(new TFormulaPrimitive("TMath::Sin","TMath::Sin",sin));              // 11
   AddFormula(new TFormulaPrimitive("sin","sin",sin));                            // 11
   AddFormula(new TFormulaPrimitive("TMath::Tan","TMath::Tan",tan));              // 12
   AddFormula(new TFormulaPrimitive("tan","tan",tan));                            // 12
   AddFormula(new TFormulaPrimitive("TMath::ACos","TMath::ACos",acos));           // 13
   AddFormula(new TFormulaPrimitive("acos","acos",acos));                         // 13
   AddFormula(new TFormulaPrimitive("TMath::ASin","TMath::ASin",asin));           // 14
   AddFormula(new TFormulaPrimitive("asin","asin",asin));                         // 14
   AddFormula(new TFormulaPrimitive("TMath::ATan","TMath::ATan",atan));           // 15
   AddFormula(new TFormulaPrimitive("atan","atan",atan));                         // 15
   AddFormula(new TFormulaPrimitive("TMath::ATan2","TMath::ATan2",atan2));        // 16
   AddFormula(new TFormulaPrimitive("atan2","atan2",atan2));                      // 16
   //   kpow      = 20, ksq = 21, ksqrt     = 22,
   AddFormula(new TFormulaPrimitive("pow","pow",TMath::Power));                 //20
   AddFormula(new TFormulaPrimitive("sq","sq",TFastFun::Pow2));                 //21
   AddFormula(new TFormulaPrimitive("sqrt","sqrt",TFastFun::Sqrt));             //22
   // kmin      = 24, kmax = 25,
   AddFormula(new TFormulaPrimitive("min","min",(GenFunc110)TMath::Min));       //24
   AddFormula(new TFormulaPrimitive("max","max",(GenFunc110)TMath::Max));       //25
   // klog      = 30, kexp = 31, klog10 = 32,
   AddFormula(new TFormulaPrimitive("log","log",TMath::Log));                           //30
   AddFormula(new TFormulaPrimitive("exp","exp",TMath::Exp));                           //31
   AddFormula(new TFormulaPrimitive("log10","log10",TMath::Log10));                     //32
   //
   //    cosh        70                  acosh        73
   //    sinh        71                  asinh        74
   //    tanh        72                  atanh        75
   //
   AddFormula(new TFormulaPrimitive("TMath::CosH","TMath::Cosh",cosh));                     // 70
   AddFormula(new TFormulaPrimitive("cosh","cosh",cosh));                                   // 70
   AddFormula(new TFormulaPrimitive("TMath::SinH","TMath::SinH",sinh));                     // 71
   AddFormula(new TFormulaPrimitive("sinh","sinh",sinh));                                   // 71
   AddFormula(new TFormulaPrimitive("TMath::TanH","TMath::Tanh",tanh));                     // 72
   AddFormula(new TFormulaPrimitive("tanh","tanh",tanh));                                   // 72
   AddFormula(new TFormulaPrimitive("TMath::ACosH","TMath::ACosh",TMath::ACosH));           // 73
   AddFormula(new TFormulaPrimitive("acosh","acosH",TMath::ACosH));                         // 73
   AddFormula(new TFormulaPrimitive("TMath::ASinH","TMath::ASinh",TMath::ASinH));           // 74
   AddFormula(new TFormulaPrimitive("acosh","acosH",TMath::ASinH));                         // 74
   AddFormula(new TFormulaPrimitive("TMath::ATanH","TMath::ATanh",TMath::ATanH));           // 75
   AddFormula(new TFormulaPrimitive("atanh","atanh",TMath::ATanH));                         // 75
   //
   AddFormula(new TFormulaPrimitive("TMath::Abs","TMath::Abs",TMath::Abs));
   AddFormula(new TFormulaPrimitive("TMath::BreitWigner","TMath::BreitWigner",TMath::BreitWigner));

   //Disable direct access to TMath::Landau for now because of the default parameter.
   //AddFormula(new TFormulaPrimitive("TMath::Landau","TMath::Landau",(TFormulaPrimitive::GenFunc1110)TMath::Landau));

   TMath_GenerInterface();
   return 1;
}

   } // end namespace v5

} // end namespace ROOT
