// @(#)root/hist:$Id$
// Author: Marian Ivanov, 2005

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include <math.h>

#include "TFormulaOldPrimitive.h"
#include "TMath.h"
#include "TVirtualMutex.h"
#ifdef WIN32
#pragma optimize("",off)
#endif

static TVirtualMutex* gTFormulaOldPrimativeListMutex = 0;

void TMath_GenerInterface();

ClassImp(TFormulaOldPrimitive)

//______________________________________________________________________________
// The Formula Primitive class
//
//    Helper class for TFormulaOld to speed up TFormulaOld evaluation
//    TFormulaOld can use all functions registered in the list of TFormulaOldPrimitives
//    User can add new function to the list of primitives
//    if FormulaPrimitive with given name is already defined new primitive is ignored
//    Example:
//      TFormulaOldPrimitive::AddFormula(new TFormulaOldPrimitive("Pow2","Pow2",TFastFun::Pow2));
//      TF1 f1("f1","Pow2(x)");
//
//
//
//    TFormulaOldPrimitive is used to get direct acces to the function pointers
//    GenFunc     -  pointers  to the static function
//    TFunc       -  pointers  to the data member functions
//
//    The following sufixes are currently used, to describe function arguments:
//    ------------------------------------------------------------------------
//    G     - generic layout - pointer to double (arguments), pointer to double (parameters)
//    10    - double
//    110   - double, double
//    1110  - double, double, double


//______________________________________________________________________________
// TFormulaOld primitive
//
TObjArray * TFormulaOldPrimitive::fgListOfFunction = 0;
#ifdef R__COMPLETE_MEM_TERMINATION
namespace {
   class TFormulaOldPrimitiveCleanup {
      TObjArray **fListOfFunctions;
   public:
      TFormulaOldPrimitiveCleanup(TObjArray **functions) : fListOfFunctions(functions) {}
      ~TFormulaOldPrimitiveCleanup() {
         delete *fListOfFunctions;
      }
   };

}
#endif

//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive() : TNamed(),
                                         fFuncG(0),
                                         fType(0),fNArguments(0),fNParameters(0),fIsStatic(kTRUE)
{
   // Default constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     GenFunc0 fpointer) : TNamed(name,formula),
                                                          fFunc0(fpointer),
                                                          fType(0),fNArguments(0),fNParameters(0),fIsStatic(kTRUE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     GenFunc10 fpointer) : TNamed(name,formula),
                                                           fFunc10(fpointer),
                                                           fType(10),fNArguments(1),fNParameters(0),fIsStatic(kTRUE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     GenFunc110 fpointer) : TNamed(name,formula),
                                                            fFunc110(fpointer),
                                                            fType(110),fNArguments(2),fNParameters(0),fIsStatic(kTRUE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     GenFunc1110 fpointer) : TNamed(name,formula),
                                                             fFunc1110(fpointer),
                                                             fType(1110),fNArguments(3),fNParameters(0),fIsStatic(kTRUE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     GenFuncG fpointer,Int_t npar) : TNamed(name,formula),
                                                                     fFuncG(fpointer),
                                                                     fType(-1),fNArguments(2),fNParameters(npar),fIsStatic(kTRUE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     TFuncG fpointer) : TNamed(name,formula),
                                                        fTFuncG(fpointer),
                                                        fType(0),fNArguments(0),fNParameters(0),fIsStatic(kFALSE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     TFunc0 fpointer) : TNamed(name,formula),
                                                        fTFunc0(fpointer),
                                                        fType(0),fNArguments(0),fNParameters(0),fIsStatic(kFALSE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     TFunc10 fpointer) : TNamed(name,formula),
                                                         fTFunc10(fpointer),
                                                         fType(-10),fNArguments(1),fNParameters(0),fIsStatic(kFALSE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     TFunc110 fpointer) : TNamed(name,formula),
                                                          fTFunc110(fpointer),
                                                          fType(-110),fNArguments(2),fNParameters(0),fIsStatic(kFALSE)
{
   // Constructor.

}


//______________________________________________________________________________
TFormulaOldPrimitive::TFormulaOldPrimitive(const char *name,const char *formula,
                                     TFunc1110 fpointer) :TNamed(name,formula),
                                                          fTFunc1110(fpointer),
                                                          fType(-1110),fNArguments(3),fNParameters(0),fIsStatic(kFALSE)
{
   // Constructor.

}


//______________________________________________________________________________
Int_t TFormulaOldPrimitive::AddFormula(TFormulaOldPrimitive * formula)
{
   // Add formula to the list of primitive formulas.
   // If primitive formula already defined do nothing.
   R__LOCKGUARD2(gTFormulaOldPrimativeListMutex);
   if (fgListOfFunction == 0) BuildBasicFormulas();
   if (FindFormula(formula->GetName(),formula->fNArguments)){
      delete formula;
      return 0;
   }
   fgListOfFunction->AddLast(formula);
   return 1;
}


//______________________________________________________________________________
Double_t TFormulaOldPrimitive::Eval(Double_t* x)
{
   // Eval primitive function at point x.

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


//______________________________________________________________________________
Double_t  TFormulaOldPrimitive::Eval(TObject *o, Double_t *x)
{
   // Eval member function of object o at point x.

   if (fIsStatic == kTRUE) return 0;
   if (fType== 0)    return (*o.*fTFunc0)();
   if (fType==-10)   return (*o.*fTFunc10)(*x);
   if (fType==-110)  return (*o.*fTFunc110)(x[0],x[1]);
   if (fType==-1110) return (*o.*fTFunc1110)(x[0],x[1],x[2]);
   return 0;
}


//______________________________________________________________________________
Double_t TFormulaOldPrimitive::Eval(Double_t *x, Double_t *param)
{
   // Eval primitive parametric function.

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
   // Namespace with basic primitive functions registered by TFormulaOldPrimitive
   // all function registered by TFormulaOldPrimitive can be used in TFormulaOld
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


//______________________________________________________________________________
TFormulaOldPrimitive* TFormulaOldPrimitive::FindFormula(const char* name)
{
   // Find the formula in the list of formulas.
   R__LOCKGUARD2(gTFormulaOldPrimativeListMutex);
   if (!fgListOfFunction) {
      BuildBasicFormulas();
   }
   Int_t nobjects = fgListOfFunction->GetEntries();
   for (Int_t i = 0; i < nobjects; ++i) {
      TFormulaOldPrimitive *formula = (TFormulaOldPrimitive*)fgListOfFunction->At(i);
      if (formula && 0==strcmp(name, formula->GetName())) return formula;
   }
   return 0;
}


//______________________________________________________________________________
TFormulaOldPrimitive* TFormulaOldPrimitive::FindFormula(const char* name, UInt_t nargs)
{
   // Find the formula in the list of formulas.

   R__LOCKGUARD2(gTFormulaOldPrimativeListMutex);
   if (!fgListOfFunction) {
      BuildBasicFormulas();
   }
   Int_t nobjects = fgListOfFunction->GetEntries();
   for (Int_t i = 0; i < nobjects; ++i) {
      TFormulaOldPrimitive *prim = (TFormulaOldPrimitive*)fgListOfFunction->At(i);
      if (prim) {
         bool match = ( ((UInt_t)prim->fNArguments) == nargs );
         if (match && 0==strcmp(name, prim->GetName())) return prim;
      }
   }
   return 0;
}


//______________________________________________________________________________
TFormulaOldPrimitive* TFormulaOldPrimitive::FindFormula(const char* name, const char *args)
{
   // Find the formula in the list of formulas.

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


//______________________________________________________________________________
Double_t TFastFun::FPoln(Double_t *x, Double_t *param, Int_t npar)
{
   // FPoln.

   Double_t res = 0; Double_t temp=1;
   for (Int_t j=npar ;j>=0;j--) {
      res  += temp*param[j];
      temp *= *x;
   }
   return res;
}


//______________________________________________________________________________
Double_t TFastFun::Gaus(Double_t x, Double_t mean, Double_t sigma)
{
   // Gauss.

   if (sigma == 0) return 1.e30;
   Double_t arg = (x-mean)/sigma;
   return TMath::Exp(-0.5*arg*arg);
}


//______________________________________________________________________________
Double_t TFastFun::Gausn(Double_t x, Double_t mean, Double_t sigma)
{
   // Normalize gauss.

   if (sigma == 0)  return 0;
   Double_t arg = (x-mean)/sigma;
   return TMath::Exp(-0.5*arg*arg)/(2.50662827463100024*sigma);  //sqrt(2*Pi)=2.50662827463100024
}


//______________________________________________________________________________
Int_t TFormulaOldPrimitive::BuildBasicFormulas()
{
   // Built-in functions.
   R__LOCKGUARD2(gTFormulaOldPrimativeListMutex);
   if (fgListOfFunction==0) {
      fgListOfFunction = new TObjArray(1000);
      fgListOfFunction->SetOwner(kTRUE);
   }
#ifdef R__COMPLETE_MEM_TERMINATION
   static TFormulaOldPrimitiveCleanup gCleanup(&fgListOfFunction);
#endif

   //
   // logical
   //
   AddFormula(new TFormulaOldPrimitive("XandY","XandY",TFastFun::XandY));
   AddFormula(new TFormulaOldPrimitive("XorY","XorY",TFastFun::XorY));
   AddFormula(new TFormulaOldPrimitive("XNot","XNot",TFastFun::XNot));
   AddFormula(new TFormulaOldPrimitive("XlY","XlY",TFastFun::XlY));
   AddFormula(new TFormulaOldPrimitive("XleY","XleY",TFastFun::XleY));
   AddFormula(new TFormulaOldPrimitive("XgY","XgY",TFastFun::XgY));
   AddFormula(new TFormulaOldPrimitive("XgeY","XgeY",TFastFun::XgeY));
   AddFormula(new TFormulaOldPrimitive("XeY","XeY",TFastFun::XeY));
   AddFormula(new TFormulaOldPrimitive("XneY","XneY",TFastFun::XneY));
   // addition  + multiplication
   AddFormula(new TFormulaOldPrimitive("PlusXY","PlusXY",TFastFun::PlusXY));
   AddFormula(new TFormulaOldPrimitive("MinusXY","MinusXY",TFastFun::MinusXY));
   AddFormula(new TFormulaOldPrimitive("MultXY","MultXY",TFastFun::MultXY));
   AddFormula(new TFormulaOldPrimitive("DivXY","DivXY",TFastFun::DivXY));
   AddFormula(new TFormulaOldPrimitive("XpYpZ","XpYpZ",TFastFun::XpYpZ));
   AddFormula(new TFormulaOldPrimitive("XxYxZ","XxYxZ",TFastFun::XxYxZ));
   AddFormula(new TFormulaOldPrimitive("XxYpZ","XxYpZ",TFastFun::XxYpZ));
   AddFormula(new TFormulaOldPrimitive("XpYxZ","XpYxZ",TFastFun::XpYxZ));
   //
   //
   AddFormula(new TFormulaOldPrimitive("Gaus","Gaus",TFastFun::Gaus));
   AddFormula(new TFormulaOldPrimitive("Gausn","Gausn",TFastFun::Gausn));
   AddFormula(new TFormulaOldPrimitive("Landau","Landau",TFastFun::Landau));
   AddFormula(new TFormulaOldPrimitive("Landaun","Landaun",TFastFun::Landaun));
   //
   //
   // polynoms
   //
   //
   AddFormula(new TFormulaOldPrimitive("Pol0","Pol0",(GenFuncG)TFastFun::FPol0,1));
   AddFormula(new TFormulaOldPrimitive("Pol1","Pol1",(GenFuncG)TFastFun::FPol1,2));
   AddFormula(new TFormulaOldPrimitive("Pol2","Pol2",(GenFuncG)TFastFun::FPol2,3));
   AddFormula(new TFormulaOldPrimitive("Pol3","Pol3",(GenFuncG)TFastFun::FPol3,4));
   AddFormula(new TFormulaOldPrimitive("Pol4","Pol4",(GenFuncG)TFastFun::FPol4,5));
   AddFormula(new TFormulaOldPrimitive("Pol5","Pol5",(GenFuncG)TFastFun::FPol5,6));
   AddFormula(new TFormulaOldPrimitive("Pol6","Pol6",(GenFuncG)TFastFun::FPol6,7));
   AddFormula(new TFormulaOldPrimitive("Pol7","Pol7",(GenFuncG)TFastFun::FPol7,8));
   AddFormula(new TFormulaOldPrimitive("Pol8","Pol8",(GenFuncG)TFastFun::FPol8,9));
   AddFormula(new TFormulaOldPrimitive("Pol9","Pol9",(GenFuncG)TFastFun::FPol9,10));
   AddFormula(new TFormulaOldPrimitive("Pol10","Pol10",(GenFuncG)TFastFun::FPol10,11));
   //
   // pows
   AddFormula(new TFormulaOldPrimitive("Pow2","Pow2",TFastFun::Pow2));
   AddFormula(new TFormulaOldPrimitive("Pow3","Pow3",TFastFun::Pow3));
   AddFormula(new TFormulaOldPrimitive("Pow4","Pow4",TFastFun::Pow4));
   AddFormula(new TFormulaOldPrimitive("Pow5","Pow5",TFastFun::Pow5));
   //
   //
   AddFormula(new TFormulaOldPrimitive("TMath::Cos","TMath::Cos",cos));              // 10
   AddFormula(new TFormulaOldPrimitive("cos","cos",cos));                            // 10
   AddFormula(new TFormulaOldPrimitive("TMath::Sin","TMath::Sin",sin));              // 11
   AddFormula(new TFormulaOldPrimitive("sin","sin",sin));                            // 11
   AddFormula(new TFormulaOldPrimitive("TMath::Tan","TMath::Tan",tan));              // 12
   AddFormula(new TFormulaOldPrimitive("tan","tan",tan));                            // 12
   AddFormula(new TFormulaOldPrimitive("TMath::ACos","TMath::ACos",acos));           // 13
   AddFormula(new TFormulaOldPrimitive("acos","acos",acos));                         // 13
   AddFormula(new TFormulaOldPrimitive("TMath::ASin","TMath::ASin",asin));           // 14
   AddFormula(new TFormulaOldPrimitive("asin","asin",asin));                         // 14
   AddFormula(new TFormulaOldPrimitive("TMath::ATan","TMath::ATan",atan));           // 15
   AddFormula(new TFormulaOldPrimitive("atan","atan",atan));                         // 15
   AddFormula(new TFormulaOldPrimitive("TMath::ATan2","TMath::ATan2",atan2));        // 16
   AddFormula(new TFormulaOldPrimitive("atan2","atan2",atan2));                      // 16
   //   kpow      = 20, ksq = 21, ksqrt     = 22,
   AddFormula(new TFormulaOldPrimitive("pow","pow",TMath::Power));                 //20
   AddFormula(new TFormulaOldPrimitive("sq","sq",TFastFun::Pow2));                 //21
   AddFormula(new TFormulaOldPrimitive("sqrt","sqrt",TFastFun::Sqrt));             //22
   // kmin      = 24, kmax = 25,
   AddFormula(new TFormulaOldPrimitive("min","min",(GenFunc110)TMath::Min));       //24
   AddFormula(new TFormulaOldPrimitive("max","max",(GenFunc110)TMath::Max));       //25
   // klog      = 30, kexp = 31, klog10 = 32,
   AddFormula(new TFormulaOldPrimitive("log","log",TMath::Log));                           //30
   AddFormula(new TFormulaOldPrimitive("exp","exp",TMath::Exp));                           //31
   AddFormula(new TFormulaOldPrimitive("log10","log10",TMath::Log10));                     //32
   //
   //    cosh        70                  acosh        73
   //    sinh        71                  asinh        74
   //    tanh        72                  atanh        75
   //
   AddFormula(new TFormulaOldPrimitive("TMath::CosH","TMath::Cosh",cosh));                     // 70
   AddFormula(new TFormulaOldPrimitive("cosh","cosh",cosh));                                   // 70
   AddFormula(new TFormulaOldPrimitive("TMath::SinH","TMath::SinH",sinh));                     // 71
   AddFormula(new TFormulaOldPrimitive("sinh","sinh",sinh));                                   // 71
   AddFormula(new TFormulaOldPrimitive("TMath::TanH","TMath::Tanh",tanh));                     // 72
   AddFormula(new TFormulaOldPrimitive("tanh","tanh",tanh));                                   // 72
   AddFormula(new TFormulaOldPrimitive("TMath::ACosH","TMath::ACosh",TMath::ACosH));           // 73
   AddFormula(new TFormulaOldPrimitive("acosh","acosH",TMath::ACosH));                         // 73
   AddFormula(new TFormulaOldPrimitive("TMath::ASinH","TMath::ASinh",TMath::ASinH));           // 74
   AddFormula(new TFormulaOldPrimitive("acosh","acosH",TMath::ASinH));                         // 74
   AddFormula(new TFormulaOldPrimitive("TMath::ATanH","TMath::ATanh",TMath::ATanH));           // 75
   AddFormula(new TFormulaOldPrimitive("atanh","atanh",TMath::ATanH));                         // 75
   //
   AddFormula(new TFormulaOldPrimitive("TMath::Abs","TMath::Abs",TMath::Abs));
   AddFormula(new TFormulaOldPrimitive("TMath::BreitWigner","TMath::BreitWigner",TMath::BreitWigner));

   //Disable direct access to TMath::Landau for now because of the default parameter.
   //AddFormula(new TFormulaOldPrimitive("TMath::Landau","TMath::Landau",(TFormulaOldPrimitive::GenFunc1110)TMath::Landau));

   TMath_GenerInterface();
   return 1;
}
