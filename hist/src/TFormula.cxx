// @(#)root/hist:$Name:  $:$Id: TFormula.cxx,v 1.59 2003/11/25 17:07:30 brun Exp $
// Author: Nicolas Brun   19/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <math.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TClass.h"
#include "TFormula.h"
#include "TMath.h"
#include "TRandom.h"
#include "TFunction.h"
#include "TMethodCall.h"
#include "TObjString.h"

#ifdef WIN32
#pragma optimize("",off)
#endif

static Int_t MAXOP,MAXPAR,MAXCONST;
const Int_t kMAXSTRINGFOUND = 10;

ClassImp(TFormula)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*The  F O R M U L A  class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*
//*-*   This class has been implemented by begin_html <a href="http://pcbrun.cern.ch/nicolas/index.html">Nicolas Brun</a> end_html(age 18).
//*-*   ========================================================
//Begin_Html
/*
<img src="gif/tformula_classtree.gif">
*/
//End_Html
//*-*
//*-*  Example of valid expressions:
//*-*     -  sin(x)/x
//*-*     -  [0]*sin(x) + [1]*exp(-[2]*x)
//*-*     -  x + y**2
//*-*     -  x^2 + y^2
//*-*     -  [0]*pow([1],4)
//*-*     -  2*pi*sqrt(x/y)
//*-*     -  gaus(0)*expo(3) + ypol3(5)*x
//*-*
//*-*  In the last example above:
//*-*     gaus(0) is a substitute for [0]*exp(-0.5*((x-[1])/[2])**2)
//*-*        and (0) means start numbering parameters at 0
//*-*     expo(3) is a substitute for exp([3]+[4])*x)
//*-*     pol3(5) is a substitute for par[5]+par[6]*x+par[7]*x**2+par[8]*x**3
//*-*         (here Pol3 stands for Polynomial of degree 3)
//*-*
//*-*   TMath functions can be part of the expression, eg:
//*-*     -  TMath::Landau(x)*sin(x)
//*-*     -  TMath::Erf(x)
//*-*
//*-*   Comparisons operators are also supported (&&, ||, ==, <=, >=, !)
//*-*   Examples:
//*-*      sin(x*(x<0.5 || x>1))
//*-*   If the result of a comparison is TRUE, the result is 1, otherwise 0.
//*-*
//*-*   Already predefined names can be given. For example, if the formula
//*-*     TFormula old(sin(x*(x<0.5 || x>1))) one can assign a name to the formula. By default
//*-*     the name of the object = title = formula itself.
//*-*     old.SetName("old").
//*-*     then, old can be reused in a new expression.
//*-*     TFormula new("x*old") is equivalent to:
//*-*     TFormula new("x*sin(x*(x<0.5 || x>1))")
//*-*
//*-*   Up to 4 dimensions are supported (indicated by x, y, z, t)
//*-*   An expression may have 0 parameters or a list of parameters
//*-*   indicated by the sequence [par_number]
//*-*
//*-*   A graph showing the logic to compile and analyze a formula
//*-*   is shown in TFormula::Compile and TFormula::Analyze.
//*-*   Once a formula has been compiled, it can be evaluated for a given
//*-*   set of parameters. see graph in TFormula::EvalPar.
//*-*
//*-*   This class is the base class for the function classes TF1,TF2 and TF3.
//*-*   It is also used by the ntuple selection mechanism TNtupleFormula.
//*-*
//*-*
//*-*     WHY TFormula CANNOT ACCEPT A CLASS MEMBER FUNCTION ?
//*-*     ====================================================
//*-* This is a frequently asked question.
//*-* C++ is a strongly typed language. There is no way for TFormula (without
//*-* recompiling this class) to know about all possible user defined data types.
//*-* This also apply to the case of a static class function.
//*-* Because TMath is a special and frequent case, TFormula is aware
//*-* of all TMath functions.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


//______________________________________________________________________________
TFormula::TFormula(): TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*Formula default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ============================

   fNdim   = 0;
   fNpar   = 0;
   fNoper  = 0;
   fNconst = 0;
   fNumber = 0;
   fExpr   = 0;
   fOper   = 0;
   fConst  = 0;
   fParams = 0;
   fNstring= 0;
   fNames  = 0;
   fNval   = 0;
}

//______________________________________________________________________________
TFormula::TFormula(const char *name,const char *expression) :TNamed(name,expression)
{
//*-*-*-*-*-*-*-*-*-*-*Normal Formula constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ==========================

   fNdim   = 0;
   fNpar   = 0;
   fNoper  = 0;
   fNconst = 0;
   fNumber = 0;
   fExpr   = 0;
   fOper   = 0;
   fConst  = 0;
   fParams = 0;
   fNstring= 0;
   fNames  = 0;
   fNval   = 0;

   if (!expression || !*expression) {
      Error("TFormula", "expression may not be 0 or have 0 length");
      return;
   }

   //eliminate blanks in expression
   Int_t i,j,nch;
   nch = strlen(expression);
   char *expr = new char[nch+1];
   j = 0;
   for (i=0;i<nch;i++) {
      if (expression[i] == ' ') continue;
      if (i > 0 && (expression[i] == '*') && (expression[i-1] == '*')) {
         expr[j-1] = '^';
         continue;
      }
      expr[j] = expression[i]; j++;
   }
   expr[j] = 0;
   if (j) SetTitle(expr);
   delete [] expr;

   if (Compile()) return;

//*-*- Store formula in linked list of formula in ROOT

   TFormula *old = (TFormula*)gROOT->GetListOfFunctions()->FindObject(name);
   if (old) {
      gROOT->GetListOfFunctions()->Remove(old);
   }
   gROOT->GetListOfFunctions()->Add(this);
}

//______________________________________________________________________________
TFormula::TFormula(const TFormula &formula) : TNamed()
{
   fNdim   = 0;
   fNpar   = 0;
   fNoper  = 0;
   fNconst = 0;
   fNumber = 0;
   fExpr   = 0;
   fOper   = 0;
   fConst  = 0;
   fParams = 0;
   fNstring= 0;
   fNames  = 0;
   fNval   = 0;

   ((TFormula&)formula).Copy(*this);
}

//______________________________________________________________________________
TFormula& TFormula::operator=(const TFormula &rhs) 
{
   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}

//______________________________________________________________________________
TFormula::~TFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Formula default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================

   gROOT->GetListOfFunctions()->Remove(this);

   ClearFormula();
}

//______________________________________________________________________________
Bool_t TFormula::AnalyzeFunction(TString &chaine, Int_t &err, Int_t offset)
{
//*-*-*-*-*-*-*-*-*Check if the chain as function call *-*-*-*-*-*-*-*-*-*-*
//*-*              =======================================
//*-*
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*

   int i,j;

   // We have to decompose the chain is 3 potential components:
   //   namespace::functionName( args )

   Ssiz_t argStart = chaine.First('(');
   if (argStart<0) return false;

   TString functionName = chaine(0,argStart);

   // This does not support template yet (where the scope operator might be in
   // one of the template arguments
   Ssiz_t scopeEnd = functionName.Last(':');
   TString spaceName;
   if (scopeEnd>0 && functionName[scopeEnd-1]==':') {
      spaceName = functionName(0,scopeEnd-1);
      functionName.Remove(0,scopeEnd+1);
   }

   // Now we need to count and decompose the actual arguments, we could also check the type
   // of the arguments
   if (chaine[chaine.Length()-1] != ')') {
      Error("AnalyzeFunction","We thought we had a function but we dont (in %s)\n",chaine.Data());
   }

   TString args = chaine(argStart+1,chaine.Length()-2-argStart);
   TObjArray argArr;
   //fprintf(stderr,"args are '%s'\n",args.Data());

   bool inString = false;
   int paran = 0;
   int brack = 0;
   int prevComma = 0;
   int nargs = 0;
   for(i=0; i<args.Length(); i++) {
      if (args[i]=='"') inString = !inString;
      if (inString) continue;

      bool foundArg = false;
      switch(args[i]) {

         case '(': paran++; break;
         case ')': paran--; break;
         case '[': brack++; break;
         case ']': brack--; break;

         case ',': if (paran==0 && brack==0) { foundArg = true; } break;
      }
      if ((i+1)==args.Length()) {
         foundArg = true; i++;
      }
      if (foundArg) {
         TString arg = args(prevComma,i-prevComma);

         // Here we could
         //   a) check the type
         //fprintf(stderr,"found #%d arg %s\n",nargs,arg.Data());

         // We register the arg for later usage
         argArr.Add(new TObjString(arg));
         nargs++;

         prevComma = i+1;
      };
   }

   if (nargs>999) {
      err = 7;
      return false;
   }

   // Now we need to lookup the function and check its arguments.

   // We have 2 choice ... parse more and replace x, y, z and [?] by 0.0 or
   // or do the following silly thing:
   TString proto;
   for(j=0; j<nargs; j++) {
      proto += "0.0,";
   }
   if (nargs) proto.Remove(proto.Length()-1);


   TClass *ns = (spaceName.Length()) ? gROOT->GetClass(spaceName) : 0;

   TMethodCall *method = new TMethodCall();
   if (ns) {
      method->Init(ns,functionName,proto);
   } else {
      method->Init(functionName,proto);
   }

   if (method->IsValid()) {

      if (method->ReturnType() == TMethodCall::kOther) {
         /*
           Error("Compile",
               "TFormula can only call interpreted and compiled function that returns a numerical type %s returns a %s\n",
               method->GetMethodName(), method->GetMethod()->GetReturnTypeName());
         */
         err=29;

      } else {

         fFunctions.Add(method);

         // Analyze the arguments
         TIter next(&argArr);
         TObjString *objstr;
         while ( (objstr=(TObjString*)next()) ) {
            Analyze(objstr->String(),err,offset);
         }

         fExpr[fNoper] = method->GetMethod()->GetPrototype();
         fOper[fNoper] = kFunctionCall + fFunctions.GetLast()*1000 + nargs;
         fNoper++;

         return true;
      }
   }

   delete method;

   return false;
}


//______________________________________________________________________________
void TFormula::Analyze(const char *schain, Int_t &err, Int_t offset)
{
//*-*-*-*-*-*-*-*-*Analyze a sub-expression in one formula*-*-*-*-*-*-*-*-*-*-*
//*-*              =======================================
//*-*
//*-*   Expressions in one formula are recursively analyzed.
//*-*   Result of analysis is stored in the object tables.
//*-*
//*-*                  Table of function codes and errors
//*-*                  ==================================
//*-*
//*-*   * functions :
//*-*
//*-*     +           1                   pow          20
//*-*     -           2                   sq           21
//*-*     *           3                   sqrt         22
//*-*     /           4                   strstr       23
//*-*     %           5                   min          24
//*-*                                     max          25
//*-*                                     log          30
//*-*     cos         10                  exp          31
//*-*     sin         11                  log10        32
//*-*     tan         12
//*-*     acos        13                  abs          41
//*-*     asin        14                  sign         42
//*-*     atan        15                  int          43
//*-*     atan2       16
//*-*     fmod        17                  rndm         50
//*-*
//*-*     cosh        70                  acosh        73
//*-*     sinh        71                  asinh        74
//*-*     tanh        72                  atanh        75
//*-*
//*-*     expo      10xx                  gaus       20xx
//*-*     expo(0)   1000                  gaus(0)    2000
//*-*     expo(1)   1001                  gaus(1)    2001
//*-*     xexpo     10xx                  xgaus      20xx
//*-*     yexpo     11xx                  ygaus      21xx
//*-*     zexpo     12xx                  zgaus      22xx
//*-*     xyexpo    15xx                  xygaus     25xx
//*-*     yexpo(5)  1105                  ygaus(5)   2105
//*-*     xyexpo(2) 1502                  xygaus(2)  2502
//*-*
//*-*     landau      40xx
//*-*     landau(0)   4000
//*-*     landau(1)   4001
//*-*     xlandau     40xx
//*-*     ylandau     41xx
//*-*     zlandau     42xx
//*-*     xylandau    45xx
//*-*     ylandau(5)  4105
//*-*     xylandau(2) 4502
//*-*
//*-*     pol0      100xx                 pol1      101xx
//*-*     pol0(0)   10000                 pol1(0)   10100
//*-*     pol0(1)   10001                 pol1(1)   10101
//*-*     xpol0     100xx                 xpol1     101xx
//*-*     ypol0     200xx                 ypol1     201xx
//*-*     zpol0     300xx                 zpol1     301xx
//*-*     ypol0(5)  20005                 ypol1(5)  20105
//*-*
//*-*     pi          40
//*-*
//*-*     &&          60                  <            64
//*-*     ||          61                  >            65
//*-*     ==          62                  <=           66
//*-*     !=          63                  =>           67
//*-*     !           68
//*-*     ==(string)  76                  &            78
//*-*     !=(string)  77                  |            79
//*-*     <<(shift)   80                  >>(shift)    81
//*-*
//*-*   * constants (kConstants) : 
//*-*
//*-*    c0  50000      c1  50001  etc..
//*-*
//*-*   * strings (kStrings) :
//*-*
//*-*    sX  80000      (80001 to 99999) 
//*-*
//*-*   * variables (kFormulaVar) :
//*-*
//*-*     x    110000     y    110001     z    110002     t    110003
//*-*
//*-*   * parameters :
//*-*
//*-*     [1]        101
//*-*     [2]        102
//*-*     etc.
//*-*  
//*-*   * boolean optimization (kBoolOptmize) :
//*-*
//*-*     Those pseudo operation are used to implement lazy evaluation of
//*-*     && and ||.  When the left hand of the expression if false 
//*-*     (respectively true), the evaluation of the right is entirely skipped
//*-*     (since it would not change the value of the expreession).
//*-*
//*-*     &&   120011 (one operation on right) 120021 (2 operations on right)
//*-*     ||   120012 (one operation on right) 120022 (2 operations on right)
//*-*  
//*-*   * functions calls (kFunctionCall) :
//*-*
//*-*    f0 200000  f1 200001  etc..
//*-*
//*-*   * errors :
//*-*
//*-*     1  : Division By Zero
//*-*     2  : Invalid Floating Point Operation
//*-*     4  : Empty String
//*-*     5  : invalid syntax
//*-*     6  : Too many operators
//*-*     7  : Too many parameters
//*-*    10  : z specified but not x and y
//*-*    11  : z and y specified but not x
//*-*    12  : y specified but not x
//*-*    13  : z and x specified but not y
//*-*    20  : non integer value for parameter number
//*-*    21  : atan2 requires two arguments
//*-*    22  : pow requires two arguments
//*-*    23  : degree of polynomial not specified
//*-*    24  : Degree of polynomial must be positive
//*-*    25  : Degree of polynomial must be less than 20
//*-*    26  : Unknown name
//*-*    27  : Too many constants in expression
//*-*    28  : strstr requires two arguments
//*-*    29  : interpreted or compiled function have to return a numerical type
//*-*    30  : Bad numerical expression
//*-*    40  : '(' is expected
//*-*    41  : ')' is expected
//*-*    42  : '[' is expected
//*-*    43  : ']' is expected
//Begin_Html
/*
<img src="gif/analyze.gif">
*/
//End_Html
//*-*
//*-*  Special functions
//*-*  -----------------
//*-*  By default, the formula is assigned fNumber=0. However, the following 
//*-*  formula built with simple functions are assigned  fNumber:
//*-*    "gaus"    100
//*-*    "expo"    200
//*-*    "polN"    300+N
//*-*    "landau"  400
//*-*  Note that expressions like gaus(0), expo(1) will force fNumber=0
//*-*
//*-*  Warning when deriving a class from TFormula
//*-*  -------------------------------------------
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


  Int_t valeur,find,n,i,j,k,lchain,nomb,virgule,inter,nest;
  Int_t compt,compt2,compt3,compt4;
  Bool_t inString;
  Double_t vafConst;
  ULong_t vafConst2;
  Bool_t parenthese;
  TString s,chaine_error,chaine1ST;
  TString s1,s2,s3,ctemp;
  TString chaine = schain;
  TFormula *oldformula;
  Int_t modulo,plus,puiss10,puiss10bis,moins,multi,divi,puiss,et,ou,petit,grand,egal,diff,peteg,grdeg,etx,oux,rshift,lshift;
  char t;
  TString slash("/"), escapedSlash("\\/");
  Int_t inter2 = 0;
  SetNumber(0);

//*-*- Verify correct matching of parenthesis and remove unnecessary parenthesis.
//*-*  ========================================================================
  lchain = chaine.Length();
  //if (chaine(lchain-2,2) == "^2") chaine = "sq(" + chaine(0,lchain-2) + ")";
  parenthese = kTRUE;
  lchain = chaine.Length();
  while (parenthese && lchain>0 && err==0){
     compt  = 0;
     compt2 = 0;
     inString = false;
     lchain = chaine.Length();
     if (lchain==0) err=4;
     else {
       for (i=1; i<=lchain; ++i) {
         if (chaine(i-1,1) == "\"") inString = !inString;
         if (!inString) {
           if (chaine(i-1,1) == "[") compt2++;
           if (chaine(i-1,1) == "]") compt2--;
           if (chaine(i-1,1) == "(") compt++;
           if (chaine(i-1,1) == ")") compt--;
         }
         if (compt < 0) err = 40; // more open parentheses than close parentheses
         if (compt2< 0) err = 42; // more ] than [
         if (compt==0 && (i!=lchain || lchain==1)) parenthese = kFALSE;
         // if (lchain<3 && chaine(0,1)!="(" && chaine(lchain-1,1)!=")") parenthese = kFALSE;
       }
       if (compt > 0) err = 41; // more ( than )
       if (compt2> 0) err = 43; // more [ than ]
       if (parenthese) chaine = chaine(1,lchain-2);
     }
  }
  if (lchain==0) err=4; // empty string
  modulo=plus=moins=multi=divi=puiss=et=ou=petit=grand=egal=diff=peteg=grdeg=etx=oux=rshift=lshift=0;
  
//*-*- Look for simple operators
//*-*  =========================

  if (err==0) {
    compt = compt2 = compt3 = compt4 = 0;puiss10=0;puiss10bis = 0;
    inString = false;
    j = lchain;
    for (i=1;i<=lchain; i++) {
      puiss10=puiss10bis=0;
      if (i>2) {
        t = chaine[i-3];
        if (strchr("0123456789",t) && chaine[i-2] == 'e' ) puiss10 = 1;
        else if (i>3) {
          t = chaine[i-4];
          if (strchr("0123456789",t) && chaine(i-3,2) == ".e" ) puiss10 = 1;
        }
      }
      if (j>2) {
        t = chaine[j-3];
        if (strchr("0123456789",t) && chaine[j-2] == 'e' ) puiss10bis = 1;
        else if (j>3) {
          t = chaine[j-4];
          if (strchr("0123456789",t) && chaine(j-3,2) == ".e" ) puiss10bis = 1;
        }
      }
      if (chaine(i-1,1) == "\"") inString = !inString;
      if (inString) continue;
      if (chaine(i-1,1) == "[") compt2++;
      if (chaine(i-1,1) == "]") compt2--;
      if (chaine(i-1,1) == "(") compt++;
      if (chaine(i-1,1) == ")") compt--;
      if (chaine(j-1,1) == "[") compt3++;
      if (chaine(j-1,1) == "]") compt3--;
      if (chaine(j-1,1) == "(") compt4++;
      if (chaine(j-1,1) == ")") compt4--;
      if (chaine(i-1,2)=="&&" && !inString && compt==0 && compt2==0 && et==0) {et=i;puiss=0;}
      if (chaine(i-1,2)=="||" && compt==0 && compt2==0 && ou==0) {puiss10=0; ou=i;}
      if (chaine(i-1,1)=="&"  && compt==0 && compt2==0 && etx==0) {etx=i;puiss=0;}
      if (chaine(i-1,1)=="|"  && compt==0 && compt2==0 && oux==0) {puiss10=0; oux=i;}
      if (chaine(i-1,2)==">>" && compt==0 && compt2==0 && rshift==0) {puiss10=0; rshift=i;}
      if (chaine(i-1,1)==">"  && compt==0 && compt2==0 && rshift==0 && grand==0)
         {puiss10=0; grand=i;}
      if (chaine(i-1,2)=="<<" && compt==0 && compt2==0 && lshift==0) {puiss10=0; lshift=i;}
      if (chaine(i-1,1)=="<"  && compt==0 && compt2==0 && lshift==0 && petit==0)
         {puiss10=0; petit=i;}
      if ((chaine(i-1,2)=="<=" || chaine(i-1,2)=="=<") && compt==0 && compt2==0
          && peteg==0) {peteg=i; puiss10=0; petit=0;}
      if ((chaine(i-1,2)=="=>" || chaine(i-1,2)==">=") && compt==0 && compt2==0
          && grdeg==0) {puiss10=0; grdeg=i; grand=0;}
      if (chaine(i-1,2) == "==" && compt == 0 && compt2 == 0 && egal == 0) {puiss10=0; egal=i;}
      if (chaine(i-1,2) == "!=" && compt == 0 && compt2 == 0 && diff == 0) {puiss10=0; diff=i;}
      if (i>1 && chaine(i-1,1) == "+" && compt == 0 && compt2 == 0 && puiss10==0) plus=i;
      if (chaine(j-1,1) == "-" && chaine(j-2,1) != "*" && chaine(j-2,1) != "/"
          && chaine(j-2,1)!="^" && compt3==0 && compt4==0 && moins==0 && puiss10bis==0) moins=j;
      if (chaine(i-1,1)=="%" && compt==0 && compt2==0 && modulo==0) {puiss10=0; modulo=i;}
      if (chaine(i-1,1)=="*" && compt==0 && compt2==0 && multi==0)  {puiss10=0; multi=i;}
      if (chaine(j-1,1)=="/" && chaine(j-2,1)!="\\"
          && compt4==0 && compt3==0 && divi==0)
        {
          puiss10=0; divi=j;
        }
      if (chaine(j-1,1)=="^" && compt4==0 && compt3==0 && puiss==0) {puiss10=0; puiss=j;}
      j--;
    }

//*-*- If operator found, analyze left and right part of the statement
//*-*  ===============================================================

    if (ou != 0) {    //check for ||
      if (ou==1 || ou==lchain-1) {
        err=5;
        chaine_error="||";
      }
      else {
        ctemp = chaine(0,ou-1);
        Analyze(ctemp.Data(),err,offset);

        fExpr[fNoper] = "|| opt";
        fOper[fNoper] = kBoolOptimize+2;
        Int_t optloc = fNoper++;

        ctemp = chaine(ou+1,lchain-ou-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "||";
        fOper[fNoper] = 61;

        fOper[optloc] += (fNoper-optloc) * 10;
        fNoper++;
      }
    } else if (et!=0) {
      if (et==1 || et==lchain-1) {
        err=5;
        chaine_error="&&";
      }
      else {
        ctemp = chaine(0,et-1);
        Analyze(ctemp.Data(),err,offset);

        fExpr[fNoper] = "&& opt";
        fOper[fNoper] = kBoolOptimize+1;
        Int_t optloc = fNoper++;

        ctemp = chaine(et+1,lchain-et-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "&&";
        fOper[fNoper] = 60;

        fOper[optloc] += (fNoper-optloc) * 10;
        fNoper++;
      }
    } else if (oux!=0) {
      if (oux==1 || oux==lchain) {
        err=5;
        chaine_error="|";
      }
      else {
        ctemp = chaine(0,oux-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(oux,lchain-oux);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "|";
        fOper[fNoper] = 79;
        fNoper++;
      }
    } else if (etx!=0) {
      if (etx==1 || etx==lchain) {
        err=5;
        chaine_error="&";
      }
      else {
        ctemp = chaine(0,etx-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(etx,lchain-etx);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "&";
        fOper[fNoper] = 78;
        fNoper++;
      }
    } else if (petit != 0) {
      if (petit==1 || petit==lchain) {
        err=5;
        chaine_error="<";
      }
      else {
        ctemp = chaine(0,petit-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(petit,lchain-petit);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "<";
        fOper[fNoper] = 64;
        fNoper++;
      }
    } else if (grand != 0) {
      if (grand==1 || grand==lchain) {
        err=5;
        chaine_error=">";
      }
      else {
        ctemp = chaine(0,grand-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(grand,lchain-grand);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = ">";
        fOper[fNoper] = 65;
        fNoper++;
      }
    } else if (peteg != 0) {
      if (peteg==1 || peteg==lchain-1) {
        err=5;
        chaine_error="<=";
      }
      else {
        ctemp = chaine(0,peteg-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(peteg+1,lchain-peteg-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "<=";
        fOper[fNoper] = 66;
        fNoper++;
      }
    } else if (grdeg != 0) {
      if (grdeg==1 || grdeg==lchain-1) {
        err=5;
        chaine_error="=>";
      }
      else {
        ctemp = chaine(0,grdeg-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(grdeg+1,lchain-grdeg-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "=>";
        fOper[fNoper] = 67;
        fNoper++;
      }
    } else if (egal != 0) {
      if (egal==1 || egal==lchain-1) {
        err=5;
        chaine_error="==";
      }
      else {
        ctemp = chaine(0,egal-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(egal+1,lchain-egal-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "==";
        fOper[fNoper] = 62;
        fNoper++;
      }
    } else if (diff != 0) {
      if (diff==1 || diff==lchain-1) {
        err=5;
        chaine_error = "!=";
      }
      else {
        ctemp = chaine(0,diff-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(diff+1,lchain-diff-1);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "!=";
        fOper[fNoper] = 63;
        fNoper++;
      }
    } else
    if (plus != 0) {
      if (plus==lchain) {
        err=5;
        chaine_error = "+";
      }
      else {
        ctemp = chaine(0,plus-1);
        Analyze(ctemp.Data(),err,offset);
        ctemp = chaine(plus,lchain-plus);
        Analyze(ctemp.Data(),err,offset);
        fExpr[fNoper] = "+";
        fOper[fNoper] = 1;
        fNoper++;
      }
      
    } else {
      if (moins != 0) {
        if (moins == 1) {
          ctemp = chaine(moins,lchain-moins);
          Analyze(ctemp.Data(),err,offset);
          fExpr[fNoper] = "-1";
          fOper[fNoper] = 0;
          fNoper++;
          fExpr[fNoper] = "-";
          fOper[fNoper] = 3;
          fNoper++;
        } else {
          if (moins == lchain) {
             err=5;
             chaine_error = "-";
          } else {
            ctemp = chaine(0,moins-1);
            Analyze(ctemp.Data(),err,offset);
            ctemp = chaine(moins,lchain-moins);
            Analyze(ctemp.Data(),err,offset);
            fExpr[fNoper] = "-";
            fOper[fNoper] = 2;
            fNoper++;
          }
        }
      } else if (modulo != 0) {
        if (modulo == 1 || modulo == lchain) {
          err=5;
          chaine_error="%";
        } else {
          ctemp = chaine(0,modulo-1);
          Analyze(ctemp.Data(),err,offset);
          ctemp = chaine(modulo,lchain-modulo);
          Analyze(ctemp.Data(),err,offset);
          fExpr[fNoper] = "%";
          fOper[fNoper] = 5;
          fNoper++;
        }
      } else if (rshift != 0) {
        if (rshift == 1 || rshift == lchain) {
          err=5;
          chaine_error=">>";
        } else {
          ctemp = chaine(0,rshift-1);
          Analyze(ctemp.Data(),err,offset);
          ctemp = chaine(rshift+1,lchain-rshift-1);
          Analyze(ctemp.Data(),err,offset);
          fExpr[fNoper] = ">>";
          fOper[fNoper] = 81;
          fNoper++;
        }
      } else if (lshift != 0) {
        if (lshift == 1 || lshift == lchain) {
          err=5;
          chaine_error=">>";
        } else {
          ctemp = chaine(0,lshift-1);
          Analyze(ctemp.Data(),err,offset);
          ctemp = chaine(lshift+1,lchain-lshift-1);
          Analyze(ctemp.Data(),err,offset);
          fExpr[fNoper] = ">>";
          fOper[fNoper] = 80;
          fNoper++;
        }
      } else {
        if (multi != 0) {
          if (multi == 1 || multi == lchain) {
            err=5;
            chaine_error="*";
          }
          else {
            ctemp = chaine(0,multi-1);
            Analyze(ctemp.Data(),err,offset);
            ctemp = chaine(multi,lchain-multi);
            Analyze(ctemp.Data(),err,offset);
            fExpr[fNoper] = "*";
            fOper[fNoper] = 3;
            fNoper++;
          }
        } else {
          if (divi != 0) {
            if (divi == 1 || divi == lchain) {
              err=5;
              chaine_error = "/";
            }
            else {
              ctemp = chaine(0,divi-1);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(divi,lchain-divi);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "/";
              fOper[fNoper] = 4;
              fNoper++;
            }
          } else {
            if (puiss != 0) {
              if (puiss == 1 || puiss == lchain) {
                err = 5;
                chaine_error = "**";
              }
              else {
                if (chaine(lchain-2,2) == "^2") {
                  ctemp = "sq(" + chaine(0,lchain-2) + ")";
                  Analyze(ctemp.Data(),err,offset);
                } else {
                  ctemp = chaine(0,puiss-1);
                  Analyze(ctemp.Data(),err,offset);
                  ctemp = chaine(puiss,lchain-puiss);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "^";
                  fOper[fNoper] = 20;
                  fNoper++;
                }
              }
            } else {

              find=0;

//*-*- Check for a numerical expression
               {
                  Bool_t hasDot = kFALSE;
                  Bool_t isHexa = kFALSE;
                  Bool_t hasExpo= kFALSE;
                  if ((chaine(0,2)=="0x")||(chaine(0,2)=="0X")) isHexa=kTRUE;
                  for (Int_t j=0; j<chaine.Length() && err==0; j++) {
                     t=chaine[j];
                     if (!isHexa) {
                        if (chaine(j,1)=="e" || chaine(j,2)=="e+" || chaine(j,2)=="e-") {
                           if (hasExpo) {
                              err=26;
                              chaine_error=chaine;
                           }
                           hasExpo = kTRUE;
                           // The previous implementation allowed a '.' in the exponent.
                           // That information was ignored (by sscanf), we now make it an error
                           // hasDot = kFALSE;
                           hasDot = kTRUE;  // forbid any additional '.'
                           if (chaine(j,2)=="e+" || chaine(j,2)=="e-") j++;
                        }
                        else {
                           if (chaine(j,1) == "." && !hasDot) hasDot = kTRUE; // accept only one '.' in the number
                           else {
                              // The previous implementation was allowing ANYTHING after the '.' and thus 
                              // made legal code like '2.3 and fpx' and was just silently ignoring the 
                              // 'and fpx'.
                              if (!strchr("0123456789",t) && (chaine(j,1)!="+" || j!=0)) {
                                 err = 30;
                                 chaine_error=chaine;
                              }
                           }
                        }
                     }
                     else {
                        if (!strchr("0123456789abcdefABCDEF",t) && (j>1)) {
                           err = 30;
                           chaine_error=chaine;
                        }
                     }
                  }
                  if (fNconst >= MAXCONST) err = 27;
                  if (!err) {
                     if (!isHexa) {if (sscanf((const char*)chaine,"%lg",&vafConst) > 0) err = 0; else err =1;}
                     else {if (sscanf((const char*)chaine,"%lx",&vafConst2) > 0) err = 0; else err=1;
                     vafConst = (Double_t) vafConst2;}
                     fExpr[fNoper] = chaine;
                     k = -1;
                     for (Int_t j=0;j<fNconst;j++) {
                        if (vafConst == fConst[j] ) k= j;
                     }
                     if ( k < 0) {  k = fNconst; fNconst++; fConst[k] = vafConst; }
                     fOper[fNoper] = kConstants + k;
                     fNoper++;
                  }
                  if (err==30) err=0;
                  else find = kTRUE;
               }


//*-*- Look for an already defined expression
              if (find==0) {
                 oldformula = (TFormula*)gROOT->GetListOfFunctions()->FindObject((const char*)chaine);
                 if (oldformula && strcmp(schain,oldformula->GetTitle())) {
                    Int_t nprior = fNpar;
                    Analyze(oldformula->GetTitle(),err,fNpar); // changes fNpar
                    fNpar = nprior;
                    find=1;
                    if (!err) {
                       Int_t npold = oldformula->GetNpar();
                       fNpar += npold;
                       for (Int_t ipar=0;ipar<npold;ipar++) {
                          fParams[ipar+fNpar-npold] = oldformula->GetParameter(ipar);
                       }
                    }
                 }
              }
              if (find == 0) {
//*-*- Check if chaine is a defined variable.
//*-*- Note that DefinedVariable can be overloaded
  	             ctemp = chaine;
                ctemp.ReplaceAll(escapedSlash, slash);
                k = DefinedVariable(ctemp);
                if (k >= 5000 && k < 10000) {
                  fExpr[fNoper] = ctemp;
                  fOper[fNoper] = kVariable + k;
                  fNstring++;
                  fNoper++;
                } else if ( k >= 0 ) {
                  fExpr[fNoper] = ctemp;
                  fOper[fNoper] = kVariable + k;
                  if (k <kMAXFOUND && !fAlreadyFound.TestBitNumber(k)) {
                     fAlreadyFound.SetBitNumber(k);
                     fNval++;
                  }
                  fNoper++;
                } else if (chaine(0,1) == "!") {
                  ctemp = chaine(1,lchain-1);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "!";
                  fOper[fNoper] = 68;
                  fNoper++;
                } else if (chaine(0,1)=="\"" && chaine(chaine.Length()-1,1)=="\"") {
                  //*-* It is a string !!!
                  fExpr[fNoper] = chaine(1,chaine.Length()-2);
                  fOper[fNoper] = kStrings;
                  fNoper++;
                } else if (chaine(0,4) == "cos(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "cos";
                  fOper[fNoper] = 10;
                  fNoper++;
                } else if (chaine(0,4) == "sin(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "sin";
                  fOper[fNoper] = 11;
                  fNoper++;
                } else if (chaine(0,4) == "tan(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "tan";
                  fOper[fNoper] = 12;
                  fNoper++;
                } else if (chaine(0,5) == "acos(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "acos";
                  fOper[fNoper] = 13;
                  fNoper++;
                } else if (chaine(0,5) == "asin(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "asin";
                  fOper[fNoper] = 14;
                  fNoper++;
                } else if (chaine(0,5) == "atan(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "atan";
                  fOper[fNoper] = 15;
                  fNoper++;
                } else if (chaine(0,5) == "cosh(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "cosh";
                  fOper[fNoper] = 70;
                  fNoper++;
                } else if (chaine(0,5) == "sinh(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "sinh";
                  fOper[fNoper] = 71;
                  fNoper++;
                } else if (chaine(0,5) == "tanh(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "tanh";
                  fOper[fNoper] = 72;
                  fNoper++;
                } else if (chaine(0,6) == "acosh(") {
                  ctemp = chaine(5,lchain-5);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "acosh";
                  fOper[fNoper] = 73;
                  fNoper++;
                } else if (chaine(0,6) == "asinh(") {
                  ctemp = chaine(5,lchain-5);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "asinh";
                  fOper[fNoper] = 74;
                  fNoper++;
                } else if (chaine(0,6) == "atanh(") {
                  ctemp = chaine(5,lchain-5);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "atanh";
                  fOper[fNoper] = 75;
                  fNoper++;
                } else if (chaine(0,3) == "sq(") {
                  ctemp = chaine(2,lchain-2);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "sq";
                  fOper[fNoper] = 21;
                  fNoper++;
                } else if (chaine(0,4) == "log(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "log";
                  fOper[fNoper] = 30;
                  fNoper++;
                } else if (chaine(0,6) == "log10(") {
                  ctemp = chaine(5,lchain-5);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "log10";
                  fOper[fNoper] = 32;
                  fNoper++;
                } else if (chaine(0,4) == "exp(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "exp";
                  fOper[fNoper] = 31;
                  fNoper++;
                } else if (chaine(0,4) == "abs(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "abs";
                  fOper[fNoper] = 41;
                  fNoper++;
                } else if (chaine(0,5) == "sign(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "sign";
                  fOper[fNoper] = 42;
                  fNoper++;
                } else if (chaine(0,4) == "int(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err,offset);
                   fExpr[fNoper] = "int";
                  fOper[fNoper] = 43;
                  fNoper++;
                } else if (chaine(0,4) == "rndm") {
                   fExpr[fNoper] = "rndm";
                  fOper[fNoper] = 50;
                  fNoper++;
                } else if (chaine(0,5) == "sqrt(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err,offset);
                  fExpr[fNoper] = "sqrt";
                  fOper[fNoper] = 22;
                  fNoper++;

//*-*- Look for an exponential
//*-*  =======================
                } else if (chaine(0,4)=="expo" || chaine(1,4)=="expo" || chaine(2,4)=="expo") {
                  chaine1ST=chaine;
                  if (chaine(1,4) == "expo") {
                    ctemp=chaine(0,1);
                    if (ctemp=="x") {
                       inter2=0;
                       if (fNdim < 1) fNdim = 1; }
                    else if (ctemp=="y") {
                            inter2=1;
                            if (fNdim < 2) fNdim = 2; }
                         else if (ctemp=="z") {
                                inter2=2;
                                if (fNdim < 3) fNdim = 3; }
                              else if (ctemp=="t") {
                                     inter2=3;
                                     if (fNdim < 4) fNdim = 4; }
                                   else {
                                      err=26; // unknown name;
                                      chaine_error=chaine1ST;
                                   }
                   chaine=chaine(1,lchain-1);
                   lchain=chaine.Length();
  // a partir d'ici indentation decalee de 4 vers la gauche
             } else inter2=0;
             if (chaine(2,4) == "expo") {
               if (chaine(0,2) != "xy") {
                   err=26; // unknown name
                   chaine_error=chaine1ST;
                   }
               else {
                   inter2=5;
                   if (fNdim < 2) fNdim = 2;
                   chaine=chaine(2,lchain-2);
                   lchain=chaine.Length();
               }
            }
            if (lchain == 4) {
                if (fNpar>=MAXPAR) err=7; // too many parameters
                if (!err) {
                   fExpr[fNoper] = chaine1ST;
                   fOper[fNoper] = 1001+100*inter2+offset;
                   if (inter2 == 5+offset && fNpar < 3+offset) fNpar = 3+offset;
                   if (fNpar < 2+offset) fNpar = 2+offset;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      if (fNpar == 2) SetNumber(200);
                   }
                }
            } else if (chaine(4,1) == "(") {
                      ctemp = chaine(5,lchain-6);
                      fExpr[fNoper] = chaine1ST;
                      for (j=0; j<ctemp.Length(); j++) {
                        t=ctemp[j];
                        if (strchr("0123456789",t)==0 && (ctemp(j,1)!="+" || j!=0)) {
                           err=20;
                           chaine_error=chaine1ST;
                        }
                      }
                      if (err==0) {
                         sscanf(ctemp.Data(),"%d",&inter);
                         if (inter>=0) {
                            inter += offset;
                            fOper[fNoper] = 1001+inter+100*inter2;
                            if (inter2 == 5) inter++;
                            if (inter+2>fNpar) fNpar = inter+2;
                            if (fNpar>=MAXPAR) err=7; // too many parameters
                            if (!err) fNoper++;
                            if (fNpar == 2) SetNumber(200);
                         } else err=20;
                      } else err = 20; // non integer value for parameter number
                    } else {
                        err=26; // unknown name
                        chaine_error=chaine;
                      }
//*-*- Look for gaus, xgaus,ygaus,xygaus
//*-*  =================================
          } else if (chaine(0,4)=="gaus" || chaine(1,4)=="gaus" || chaine(2,4)=="gaus") {
            chaine1ST=chaine;
            if (chaine(1,4) == "gaus") {
               ctemp=chaine(0,1);
               if (ctemp=="x") {
                  inter2=0;
                  if (fNdim < 1) fNdim = 1; }
               else if (ctemp=="y") {
                       inter2=1;
                       if (fNdim < 2) fNdim = 2; }
                    else if (ctemp=="z") {
                            inter2=2;
                            if (fNdim < 3) fNdim = 3; }
                         else if (ctemp=="t") {
                                 inter2=3;
                                 if (fNdim < 4) fNdim = 4; }
                              else {
                                  err=26; // unknown name
                                  chaine_error=chaine1ST;
                              }
               chaine=chaine(1,lchain-1);
               lchain=chaine.Length();
            } else inter2=0;
            if (chaine(2,4) == "gaus") {
               if (chaine(0,2) != "xy") {
                   err=26; // unknown name
                   chaine_error=chaine1ST;
                   }
               else {
                   inter2=5;
                   if (fNdim < 2) fNdim = 2;
                   chaine=chaine(2,lchain-2);
                   lchain=chaine.Length();
               }
            }
            if (lchain == 4 && err==0) {
                if (fNpar>=MAXPAR) err=7; // too many parameters
                if (!err) {
                   fExpr[fNoper] = chaine1ST;
                   fOper[fNoper] = 2001+100*inter2+offset;
                   if (inter2 == 5+offset && fNpar < 5+offset) fNpar = 5+offset;
                   if (3+offset>fNpar) fNpar = 3+offset;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      if (fNpar == 3) SetNumber(100);
                   }
                }
            } else if (chaine(4,1) == "(" && err==0) {
                      ctemp = chaine(5,lchain-6);
                      fExpr[fNoper] = chaine1ST;
                      for (j=0; j<ctemp.Length(); j++) {
                        t=ctemp[j];
                        if (strchr("0123456789",t)==0 && (ctemp(j,1)!="+" || j!=0)) {
                           err=20;
                           chaine_error=chaine1ST;
                        }
                      }
                      if (err==0) {
                          sscanf(ctemp.Data(),"%d",&inter);
                          if (inter >= 0) {
                             inter += offset;
                             fOper[fNoper] = 2001+inter+100*inter2;
                             if (inter2 == 5) inter += 2;
                             if (inter+3>fNpar) fNpar = inter+3;
                             if (fNpar>=MAXPAR) err=7; // too many parameters
                             if (!err) fNoper++;
                             if(fNpar == 3) SetNumber(100);
                         } else err = 20; // non integer value for parameter number
                      }
                   } else if (err==0) {
                       err=26; // unknown name
                       chaine_error=chaine1ST;
                     }
//*-*- Look for landau, xlandau,ylandau,xylandau
//*-*  =================================
          } else if (chaine(0,6)=="landau" || chaine(1,6)=="landau" || chaine(2,6)=="landau") {
            chaine1ST=chaine;
            if (chaine(1,6) == "landau") {
               ctemp=chaine(0,1);
               if (ctemp=="x") {
                  inter2=0;
                  if (fNdim < 1) fNdim = 1; }
               else if (ctemp=="y") {
                       inter2=1;
                       if (fNdim < 2) fNdim = 2; }
                    else if (ctemp=="z") {
                            inter2=2;
                            if (fNdim < 3) fNdim = 3; }
                         else if (ctemp=="t") {
                                 inter2=3;
                                 if (fNdim < 4) fNdim = 4; }
                              else {
                                  err=26; // unknown name
                                  chaine_error=chaine1ST;
                              }
               chaine=chaine(1,lchain-1);
               lchain=chaine.Length();
            } else inter2=0;
            if (chaine(2,6) == "landau") {
               if (chaine(0,2) != "xy") {
                   err=26; // unknown name
                   chaine_error=chaine1ST;
                   }
               else {
                   inter2=5;
                   if (fNdim < 2) fNdim = 2;
                   chaine=chaine(2,lchain-2);
                   lchain=chaine.Length();
               }
            }
            if (lchain == 6 && err==0) {
                if (fNpar>=MAXPAR) err=7; // too many parameters
                if (!err) {
                   fExpr[fNoper] = chaine1ST;
                   fOper[fNoper] = 4001+100*inter2+offset;
                   if (inter2 == 5+offset && fNpar < 5+offset) fNpar = 5+offset;
                   if (3+offset>fNpar) fNpar = 3+offset;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      if (fNpar == 3) SetNumber(400);
                   }
                }
            } else if (chaine(6,1) == "(" && err==0) {
                      ctemp = chaine(7,lchain-8);
                      fExpr[fNoper] = chaine1ST;
                      for (j=0; j<ctemp.Length(); j++) {
                        t=ctemp[j];
                        if (strchr("0123456789",t)==0 && (ctemp(j,1)!="+" || j!=0)) {
                           err=20;
                           chaine_error=chaine1ST;
                        }
                      }
                      if (err==0) {
                          sscanf(ctemp.Data(),"%d",&inter);
                          if (inter >= 0) {
                             inter += offset;
                             fOper[fNoper] = 4001+inter+100*inter2;
                             if (inter2 == 5) inter += 2;
                             if (inter+3>fNpar) fNpar = inter+3;
                             if (fNpar>=MAXPAR) err=7; // too many parameters
                             if (!err) fNoper++;
                             if (fNpar == 3) SetNumber(400);
                         } else err = 20; // non integer value for parameter number
                      }
                   } else if (err==0) {
                       err=26; // unknown name
                       chaine_error=chaine1ST;
                     }
//*-*- Look for a polynomial
//*-*  =====================
          } else if (chaine(0,3) == "pol" || chaine(1,3) == "pol") {
            chaine1ST=chaine;
            if (chaine(1,3) == "pol") {
               ctemp=chaine(0,1);
               if (ctemp=="x") {
                  inter2=1;
                  if (fNdim < 1) fNdim = 1; }
               else if (ctemp=="y") {
                       inter2=2;
                       if (fNdim < 2) fNdim = 2; }
                    else if (ctemp=="z") {
                            inter2=3;
                            if (fNdim < 3) fNdim = 3; }
                         else if (ctemp=="t") {
                                 inter2=4;
                                 if (fNdim < 4) fNdim = 4; }
                              else {
                                 err=26; // unknown name;
                                 chaine_error=chaine1ST;
                              }
               chaine=chaine(1,lchain-1);
               lchain=chaine.Length();
            } else inter2=1;
            if (chaine(lchain-1,1) == ")") {
                nomb = 0;
                for (j=3;j<lchain;j++) if (chaine(j,1)=="(" && nomb == 0) nomb = j;
                if (nomb == 3) err = 23; // degree of polynomial not specified
                if (nomb == 0) err = 40; // '(' is expected
                ctemp = chaine(nomb+1,lchain-nomb-2);
                for (j=0; j<ctemp.Length(); j++) {
                        t=ctemp[j];
                        if (strchr("0123456789",t)==0 && (ctemp(j,1)!="+" || j!=0)) {
                           err=20;
                           chaine_error=chaine1ST;
                        }
                }
                if (!err) {
                   sscanf(ctemp.Data(),"%d",&inter);
                   if (inter < 0) err = 20;
                }
            }
            else {
              nomb = lchain;
              inter = 0;
            }
            if (!err) {
              inter--;
              ctemp = chaine(3,nomb-3);
              if (sscanf(ctemp.Data(),"%d",&n) > 0) {
                 if (n < 0  ) err = 24; //Degree of polynomial must be positive
                 if (n >= 20) err = 25; //Degree of polynomial must be less than 20
              } else err = 20;
            }
            if (!err) {
              fExpr[fNoper] = chaine1ST;
              fOper[fNoper] = 10000*inter2+n*100+inter+2;
              if (inter+n+1>=fNpar) fNpar = inter + n + 2;
              if (fNpar>=MAXPAR) err=7; // too many parameters
              if (!err) {
                 fNoper++;
                 if (fNdim < 1) fNdim = 1;
                 SetNumber(300+n);
              }
            }
//*-*- Look for pow,atan2,etc
//*-*  ======================
          } else if (chaine(0,4) == "pow(") {
            compt = 4; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 22; // There are plus or minus than 2 arguments for pow
            else {
              ctemp = chaine(4,virgule-5);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "^";
              fOper[fNoper] = 20;
              fNoper++;
            }
	 } else if (chaine(0,7) == "strstr(") {
            compt = 7; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 28; // There are plus or minus than 2 arguments for strstr
            else {
              ctemp = chaine(7,virgule-8);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "strstr";
              fOper[fNoper] = 23;
              fNoper++;
            }
          } else if (chaine(0,4) == "min(") {
            compt = 4; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 22; // There are plus or minus than 2 arguments for pow
            else {
              ctemp = chaine(4,virgule-5);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "min";
              fOper[fNoper] = 24;
              fNoper++;
            }
          } else if (chaine(0,4) == "max(") {
            compt = 4; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 22; // There are plus or minus than 2 arguments for pow
            else {
              ctemp = chaine(4,virgule-5);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "max";
              fOper[fNoper] = 25;
              fNoper++;
            }

          } else if (chaine(0,6) == "atan2(") {
            compt = 6; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 21;  //{ There are plus or minus than 2 arguments for atan2
            else {
              ctemp = chaine(6,virgule-7);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "atan2";
              fOper[fNoper] = 16;
              fNoper++;
            }
          } else if (chaine(0,5) == "fmod(") {
            compt = 5; nomb = 0; virgule = 0; nest=0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == "(") nest++;
              else if (chaine(compt-1,1) == ")") nest--;
              else if (chaine(compt-1,1) == "," && nest==0) {
                nomb++;
                if (nomb == 1 && virgule == 0) virgule = compt;
              }
            }
            if (nomb != 1) err = 21;  //{ There are plus or minus than 2 arguments for fmod
            else {
              ctemp = chaine(5,virgule-6);
              Analyze(ctemp.Data(),err,offset);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err,offset);
              fExpr[fNoper] = "fmod";
              fOper[fNoper] = 17;
              fNoper++;
            }
          } else if (AnalyzeFunction(chaine,err,offset) || err) { // The '||err' is to grab an error coming from AnalyzeFunction
             if (err) {
                chaine_error = chaine;
             } else {
                // We have a function call. Note that all the work was already,
                // eventually done in AnalyzeFuntion
                //fprintf(stderr,"We found a foreign function in %s\n",chaine.Data());
             }
          } else if (chaine(0,1) == "[" && chaine(lchain-1,1) == "]") {
            fExpr[fNoper] = chaine;
            fNoper++;
            ctemp = chaine(1,lchain-2);
            for (j=0; j<ctemp.Length(); j++) {
                t=ctemp[j];
                if (strchr("0123456789",t)==0 && (ctemp(j,1)!="+" || j!=0)) {
                   err=20;
                   chaine_error=chaine1ST; // le numero ? de par[?] n'est pas un entier }
                }
            }
            if (!err) {
              sscanf(ctemp.Data(),"%d",&valeur);
              fOper[fNoper-1] = valeur + 101 + offset;
              fExpr[fNoper-1] = "[";
              fExpr[fNoper-1] = (fExpr[fNoper-1] + (long int)(valeur+offset)) + "]";
            }
          } else if (chaine == "pi") {
            fExpr[fNoper] = "pi";
            fOper[fNoper] = 40;
            fNoper++;
          }
          else {
//*-*- None of the above. 
//*-*  ==================
             err = 30;
          }

              } // because of the indentation skips above this is slightly off
            }
          }
        }
      }
    }
    
//   Test  * si y existe :  que x existe
//         * si z existe :  que x et y existent

    nomb = 1;
    for (i=1; i<=fNoper; i++) {
       if (fOper[i-1] == 97 && nomb > 0) nomb *= -1;
       if (fOper[i-1] == 98 && TMath::Abs(nomb) != 2) nomb *= 2;
       if (fOper[i-1] == 99 && TMath::Abs(nomb) != 20 && TMath::Abs(nomb) != 10) nomb *= 10;
    }
    if (nomb == 10)  err = 10; //{variable z sans x et y }
    if (nomb == 20)  err = 11; //{variables z et y sans x }
    if (nomb == 2)   err = 12; //{variable y sans x }
    if (nomb == -10) err = 13; //{variables z et x sans y }
    //*-*- Overflows
    if (fNoper>=MAXOP) err=6; // too many operators

  }

//*-*- errors!
  if (err>1) {
     cout<<endl<<"*ERROR "<<err<<" : "<<endl;
     switch(err) {
        case  2 : cout<<" Invalid Floating Point Operation"<<endl; break;
        case  4 : cout<<" Empty String"<<endl; break;
        case  5 : cout<<" Invalid Syntax \""<<(const char*)chaine_error<<"\""<<endl; break;
        case  6 : cout<<" Too many operators !"<<endl; break;
        case  7 : cout<<" Too many parameters !"<<endl; break;
        case 10 : cout<<" z specified but not x and y"<<endl; break;
        case 11 : cout<<" z and y specified but not x"<<endl; break;
        case 12 : cout<<" y specified but not x"<<endl; break;
        case 13 : cout<<" z and x specified but not y"<<endl; break;
        case 20 : cout<<" Non integer value for parameter number : "<<(const char*)chaine_error<<endl; break;
        case 21 : cout<<" ATAN2 requires two arguments"<<endl; break;
        case 22 : cout<<" POW requires two arguments"<<endl; break;
        case 23 : cout<<" Degree of polynomial not specified"<<endl; break;
        case 24 : cout<<" Degree of polynomial must be positive"<<endl; break;
        case 25 : cout<<" Degree of polynomial must be less than 20"<<endl; break;
        case 26 : cout<<" Unknown name : \""<<(const char*)chaine_error<<"\""<<endl; break;
        case 27 : cout<<" Too many constants in expression"<<endl; break;
        case 28 : cout<<" strstr requires tow arguments"<<endl; break;
        case 29 : cout<<" TFormula can only call interpreted and compiled functions that return a numerical type: \n"
                      <<chaine_error<<endl; break;
        case 30 : cout<<" Bad numerical expression : \""<<(const char*)chaine_error<<"\""<<endl; break;
        case 40 : cout<<" '(' is expected"<<endl; break;
        case 41 : cout<<" ')' is expected"<<endl; break;
        case 42 : cout<<" '[' is expected"<<endl; break;
        case 43 : cout<<" ']' is expected"<<endl; break;
     }
     err=1;
  }
  
}

//______________________________________________________________________________
void TFormula::Clear(Option_t * /*option*/ )
{
//*-*-*-*-*-*-*-*-*Resets the objects*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================
//*-*
//*-* Resets the object to its state before compilation.
//*-*

   ClearFormula();
}

//______________________________________________________________________________
void TFormula::ClearFormula(Option_t * /*option*/ )
{
//*-*-*-*-*-*-*-*-*Resets the objects*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================
//*-*
//*-* Resets the object to its state before compilation.
//*-*

   fNdim   = 0;
   fNpar   = 0;
   fNoper  = 0;
   fNconst = 0;
   fNumber = 0;
   fNstring= 0;
   fNval   = 0;

   if (fExpr)   { delete [] fExpr;   fExpr   = 0;}
   if (fOper)   { delete [] fOper;   fOper   = 0;}
   if (fConst)  { delete [] fConst;  fConst  = 0;}
   if (fParams) { delete [] fParams; fParams = 0;}
   if (fNames)  { delete [] fNames;  fNames  = 0;}
   fFunctions.Delete();

   // should we also remove the object from the list?
   // gROOT->GetListOfFunctions()->Remove(this);
   // if we don't, what happens if it fails the new compilation?
}

//______________________________________________________________________________
Int_t TFormula::Compile(const char *expression)
{
//*-*-*-*-*-*-*-*-*-*-*Compile expression already stored in fTitle*-*-*-*-*-*
//*-*                  ===========================================
//*-*
//*-*   Loop on all subexpressions of formula stored in fTitle
//*-*
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*
//Begin_Html
/*
<img src="gif/compile.gif">
*/
//End_Html
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  Int_t i,j,lc,valeur,err;
  TString ctemp;

  ClearFormula();

//*-*- If expression is not empty, take it, otherwise take the title
  if (strlen(expression)) SetTitle(expression);

  TString chaine = GetTitle();
//  chaine.ToLower();

  MAXOP   = 1000;
  MAXPAR  = 100;
  MAXCONST= 100;

  fExpr   = new TString[MAXOP];
  fOper   = new Int_t[MAXOP];
  fConst  = new Double_t[MAXCONST];
  fParams = new Double_t[MAXPAR];
  fNames  = new TString[MAXPAR];
  for (i=0; i<MAXPAR; i++) {
      fParams[i] = 0;
      fNames[i] = "";
  }
  for (i=0; i<MAXOP; i++) {
      fExpr[i] = "";
      fOper[i] = 0;
  }
  for (i=0; i<MAXCONST; i++)
      fConst[i] = 0;

//*-*- Substitution of some operators to C++ style
//*-*  ===========================================
  for (i=1; i<=chaine.Length(); i++) {
    lc =chaine.Length();
    if (chaine(i-1,2) == "**") {
       chaine = chaine(0,i-1) + "^" + chaine(i+1,lc-i-1);
       i=0;
    } else
       if (chaine(i-1,2) == "++") {
          chaine = chaine(0,i) + chaine(i+1,lc-i-1);
          i=0;
       } else
          if (chaine(i-1,2) == "+-" || chaine(i-1,2) == "-+") {
             chaine = chaine(0,i-1) + "-" + chaine(i+1,lc-i-1);
             i=0;
          } else
             if (chaine(i-1,2) == "--") {
                chaine = chaine(0,i-1) + "+" + chaine(i+1,lc-i-1);
                i=0;
             } else
               if (chaine(i-1,2) == "->") {
                  chaine = chaine(0,i-1) + "." + chaine(i+1,lc-i-1);
                  i=0;
               } else
                  if (chaine(i-1,1) == "[") {
                     for (j=1;j<=chaine.Length()-i;j++) {
                        if (chaine(j+i-1,1) == "]" || j+i > chaine.Length()) break;
                     }
                     ctemp = chaine(i,j-1);
                     valeur=0;
                     sscanf(ctemp.Data(),"%d",&valeur);
                     if (valeur >= fNpar) fNpar = valeur+1;
                  } else
                     if (chaine(i-1,1) == " ") {
                       chaine = chaine(0,i-1)+chaine(i,lc-i);
                       i=0;
                     }
  }
  err = 0;
  Analyze((const char*)chaine,err);

//*-*- if no errors, copy local parameters to formula objects
  if (!err) {
     if (fNdim <= 0) fNdim = 1;
     if (chaine.Length() > 4 && GetNumber() != 400) SetNumber(0);
     //*-*- if formula is a gaussian, set parameter names
     if (GetNumber() == 100) {
        SetParName(0,"Constant");
        SetParName(1,"Mean");
        SetParName(2,"Sigma");
     }
     //*-*- if formula is an exponential, set parameter names
     if (GetNumber() == 200) {
        SetParName(0,"Constant");
        SetParName(1,"Slope");
     }
     //*-*- if formula is a polynome, set parameter names
     if (GetNumber() == 300+fNpar) {
        for (i=0;i<fNpar;i++) SetParName(i,Form("p%d",i));
     }
     //*-*- if formula is a landau, set parameter names
     if (GetNumber() == 400) {
        SetParName(0,"Constant");
        SetParName(1,"MPV");
        SetParName(2,"Sigma");
     }
  }


//*-* replace 'normal' == or != by ==(string) or !=(string) if needed.
  Int_t is_it_string,last_string=0,before_last_string=0;
  if (!fOper) fNoper = 0;
   enum { kIsCharacter = BIT(12) };
  for (i=0; i<fNoper; i++,
                      before_last_string = last_string,
                      last_string = is_it_string) {
     is_it_string = IsString(i);
     if (is_it_string) continue;
     if (last_string) {
        if (fOper[i] == 62) {
           if (!before_last_string) {
              Error("Compile", "Both operands of the operator == have to be either numbers or strings");
              return -1;
           }
           fOper[i] = 76;
           SetBit(kIsCharacter);
        } else if (fOper[i] == 63) {
           if (!before_last_string) {
              Error("Compile", "Both operands of the operator != have to be either numbers or strings");
              return -1;
           }
           fOper[i] = 77;
           SetBit(kIsCharacter);
        }
        else if (fOper[i] == 23) {
           if (! (before_last_string && last_string) ) {
              Error("Compile", "strstr requires 2 string arguments");
              return -1;
           }
           SetBit(kIsCharacter);
        } else if (before_last_string) {
           // the i-2 element is a string not used in a string operation, let's down grade it
           // to a char array:
           if (fOper[i-2]>=105000) {
              fOper[i-2] -= 5000;
              fNval++;
              fNstring--;
           }
        }
        
     } else if (before_last_string) {
        // the i-2 element is a string not used in a string operation, let's down grade it
        // to a char array:
        if (fOper[i-2]>=105000){
           fOper[i-2] -= 5000;
           fNval++;
           fNstring--;
        }
     }
  }

  if (err) { fNdim = 0; return 1; }
  return 0;
}

//______________________________________________________________________________
void TFormula::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   Int_t i;
   obj.Clear(); // delete existing arrays in the copy.
   TNamed::Copy(obj);
   ((TFormula&)obj).fNdim   = fNdim;
   ((TFormula&)obj).fNpar   = fNpar;
   ((TFormula&)obj).fNoper  = fNoper;
   ((TFormula&)obj).fNconst = fNconst;
   ((TFormula&)obj).fNumber = fNumber;
   ((TFormula&)obj).fNval   = fNval;
   ((TFormula&)obj).fExpr   = 0;
   ((TFormula&)obj).fOper   = 0;
   ((TFormula&)obj).fConst  = 0;
   ((TFormula&)obj).fParams = 0;
   ((TFormula&)obj).fNames  = 0;
   if (fExpr && fNoper) {
      ((TFormula&)obj).fExpr = new TString[fNoper];
      for (i=0; i<fNoper; i++)
         ((TFormula&)obj).fExpr[i] = "";
   }
   if (fOper && fNoper) {
      ((TFormula&)obj).fOper = new Int_t[fNoper];
      for (i=0; i<fNoper; i++)
         ((TFormula&)obj).fOper[i] = 0;
   }
   if (fConst && fNconst) {
      ((TFormula&)obj).fConst = new Double_t[fNconst];
      for (i=0; i<fNconst; i++)
         ((TFormula&)obj).fConst[i] = 0;
   }
   if (fParams && fNpar) {
      ((TFormula&)obj).fParams = new Double_t[fNpar];
      for (i=0; i<fNpar; i++)
         ((TFormula&)obj).fParams[i] = 0;
   }
   if (fNames && fNpar) {
      ((TFormula&)obj).fNames = new TString[fNpar];
      for (i=0; i<fNpar; i++)
         ((TFormula&)obj).fNames[i] = "";
   }
   for (i=0;i<fNoper;i++)  ((TFormula&)obj).fExpr[i]   = fExpr[i];
   for (i=0;i<fNoper;i++)  ((TFormula&)obj).fOper[i]   = fOper[i];
   for (i=0;i<fNconst;i++) ((TFormula&)obj).fConst[i]  = fConst[i];
   for (i=0;i<fNpar;i++)   ((TFormula&)obj).fParams[i] = fParams[i];
   for (i=0;i<fNpar;i++)   ((TFormula&)obj).fNames[i]  = fNames[i];

   TIter next(&fFunctions);
   TObject *fobj;
   while ( (fobj = next()) ) {
      ((TFormula&)obj).fFunctions.Add( fobj->Clone() );
   }
}

//______________________________________________________________________________
char *TFormula::DefinedString(Int_t)
{
//*-*-*-*-*-*Return address of string corresponding to special code*-*-*-*-*-*
//*-*        ======================================================
//*-*
//*-*   This member function is inactive in the TFormula class.
//*-*   It may be redefined in derived classes.
//*-*
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*
///*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  return 0;
}

//______________________________________________________________________________
Double_t TFormula::DefinedValue(Int_t)
{
//*-*-*-*-*-*Return value corresponding to special code*-*-*-*-*-*-*-*-*
//*-*        ==========================================
//*-*
//*-*   This member function is inactive in the TFormula class.
//*-*   It may be redefined in derived classes.
//*-*
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*
///*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  return 0;
}

//______________________________________________________________________________
Int_t TFormula::DefinedVariable(TString &chaine)
{
//*-*-*-*-*-*Check if expression is in the list of defined variables*-*-*-*-*
//*-*        =======================================================
//*-*
//*-*   This member function can be overloaded in derived classes
//*-*
//*-*   If you overload this member function, you also HAVE TO
//*-*   never call the constructor:
//*-*
//*-*     TFormula::TFormula(const char *name,const char *expression)
//*-*
//*-*   and write your own constructor
//*-*
//*-*     MyClass::MyClass(const char *name,const char *expression) : TFormula()
//*-*
//*-*   which has to call the TFormula default constructor and whose implementation
//*-*   should be similar to the implementation of the normal TFormula constructor
//*-*
//*-*   This is necessary because the normal TFormula constructor call indirectly
//*-*   the virtual member functions Analyze, DefaultString, DefaultValue
//*-*   and DefaultVariable.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  if (chaine == "x") {
     if (fNdim < 1) fNdim = 1;
     return 10000;
  } else if (chaine == "y") {
     if (fNdim < 2) fNdim = 2;
     return 10001;
  } else if (chaine == "z") {
     if (fNdim < 3) fNdim = 3;
     return 10002;
  } else if (chaine == "t") {
     if (fNdim < 4) fNdim = 4;
     return 10003;
  }
  return -1;
}

//______________________________________________________________________________
Double_t TFormula::Eval(Double_t x, Double_t y, Double_t z, Double_t t)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
//*-*
//*-*   The current value of variables x,y,z,t is passed through x, y, z and t.
//*-*   The parameters used will be the ones in the array params if params is given
//*-*    otherwise parameters will be taken from the stored data members fParams
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  Double_t xx[4];
  xx[0] = x;
  xx[1] = y;
  xx[2] = z;
  xx[3] = t;
  return EvalPar(xx);
}

//______________________________________________________________________________
Double_t TFormula::EvalPar(const Double_t *x, const Double_t *params)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
//*-*
//*-*   The current value of variables x,y,z,t is passed through the pointer x.
//*-*   The parameters used will be the ones in the array params if params is given
//*-*    otherwise parameters will be taken from the stored data members fParams
//Begin_Html
/*
<img src="gif/eval.gif">
*/
//End_Html
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  Int_t i,j,pos,pos2,inter,inter2,int1,int2;
  Float_t aresult;
  Double_t tab[kMAXFOUND];
  char *tab2[kMAXSTRINGFOUND];
  Double_t param_calc[kMAXFOUND];
  Double_t dexp,intermede,intermede1,intermede2;
  char *string_calc[kMAXSTRINGFOUND];
  Int_t precalculated = 0;
  Int_t precalculated_str = 0;

  if (params) {
     for (j=0;j<fNpar;j++) fParams[j] = params[j];
  }
  pos  = 0;
  pos2 = 0;
  for (i=0; i<fNoper; i++) {
    Int_t action = fOper[i];
//*-*- an external function call
    if (action >= kFunctionCall) {
       int fno   = (action-kFunctionCall) / 1000;
       int nargs = (action-kFunctionCall) % 1000;

       // Retrieve the function
       TMethodCall *method = (TMethodCall*)fFunctions.At(fno);

       // Set the arguments
       TString args;
       if (nargs) {
          UInt_t argloc = pos-nargs;
          for(j=0;j<nargs;j++,argloc++,pos--) {
             if (TMath::IsNaN(tab[argloc])) {
                // TString would add 'nan' this is not what we want
                // so let's do somethign else
                args += "(double)(0x8000000000000)";
             } else {
                args += tab[argloc];
             }
             args += ',';
          }
          args.Remove(args.Length()-1);
       }
       pos++;
       Double_t ret;
       method->Execute(args,ret);
       tab[pos-1] = ret; // check for the correct conversion!

       continue;
    }
//*-*- boolean operation optimizer
    if (action >= kBoolOptimize) {
       Bool_t skip = kFALSE;
       int op = (action-kBoolOptimize) % 10; // 1 is && , 2 is ||
       
       if (op == 1 && (!tab[pos-1]) ) { 
          // &&: skip the right part if the left part is already false
          
          skip = kTRUE;
          
          // Preserve the existing behavior (i.e. the result of a&&b is
          // either 0 or 1)
          tab[pos-1] = 0;
          
       } else if (op == 2 && tab[pos-1] ) {  
          // ||: skip the right part if the left part is already true
          
          skip = kTRUE;
          
          // Preserve the existing behavior (i.e. the result of a||b is
          // either 0 or 1)
          tab[pos-1] = 1;
       }
       
       if (skip) {
          int toskip = (action-kBoolOptimize) / 10;
          i += toskip;
       }
       continue;
    }
//*-*- a variable
    if (action >= kFormulaVar) {
       pos++; tab[pos-1] = x[action-kFormulaVar];
       continue;
    }
//*-*- a tree string
    if (action >= 105000) {
       if (!precalculated_str) {
          precalculated_str=1;
          for (j=0;j<fNstring;j++) string_calc[j]=DefinedString(j);
       }
       pos2++; tab2[pos2-1] = string_calc[action-105000];
       continue;
    }
//*-*- a tree variable
    if (action >= kVariable) {
       if (!precalculated) {
          precalculated = 1;
          for(j=0;j<fNval;j++) param_calc[j]=DefinedValue(j);
       }
       pos++; tab[pos-1] = param_calc[action-kVariable];
       continue;
    }
//*-*- String
    if (action == kStrings) {
       pos2++; tab2[pos2-1] = (char*)fExpr[i].Data();
       continue;
    }
//*-*- numerical value
    if (action >= kConstants) {
       pos++; tab[pos-1] = fConst[action-kConstants];
       continue;
    }
    if (action == 0) {
      pos++;
      sscanf((const char*)fExpr[i],"%g",&aresult);
      tab[pos-1] = aresult;
//*-*- basic operators and mathematical library
    } else if (action < 100) {
        switch(action) {
          case   1 : pos--; tab[pos-1] += tab[pos]; break;
          case   2 : pos--; tab[pos-1] -= tab[pos]; break;
          case   3 : pos--; tab[pos-1] *= tab[pos]; break;
          case   4 : pos--; if (tab[pos] == 0) tab[pos-1] = 0; //  division by 0
                            else               tab[pos-1] /= tab[pos];
                     break;
          case   5 : {pos--; int1=Int_t(tab[pos-1]); int2=Int_t(tab[pos]); tab[pos-1] = Double_t(int1%int2); break;}
          case  10 : tab[pos-1] = TMath::Cos(tab[pos-1]); break;
          case  11 : tab[pos-1] = TMath::Sin(tab[pos-1]); break;
          case  12 : if (TMath::Cos(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                     else tab[pos-1] = TMath::Tan(tab[pos-1]);
                     break;
          case  13 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ACos(tab[pos-1]);
                     break;
          case  14 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ASin(tab[pos-1]);
                     break;
          case  15 : tab[pos-1] = TMath::ATan(tab[pos-1]); break;
          case  70 : tab[pos-1] = TMath::CosH(tab[pos-1]); break;
          case  71 : tab[pos-1] = TMath::SinH(tab[pos-1]); break;
          case  72 : if (TMath::CosH(tab[pos-1]) == 0) {tab[pos-1] = 0;} // { tangente indeterminee }
                     else tab[pos-1] = TMath::TanH(tab[pos-1]);
                     break;
          case  73 : if (tab[pos-1] < 1) {tab[pos-1] = 0;} //  indetermination
                     else tab[pos-1] = TMath::ACosH(tab[pos-1]);
                     break;
          case  74 : tab[pos-1] = TMath::ASinH(tab[pos-1]); break;
          case  75 : if (TMath::Abs(tab[pos-1]) > 1) {tab[pos-1] = 0;} // indetermination
                     else tab[pos-1] = TMath::ATanH(tab[pos-1]); break;
          case  16 : pos--; tab[pos-1] = TMath::ATan2(tab[pos-1],tab[pos]); break;
          case  17 : pos--; tab[pos-1] = fmod(tab[pos-1],tab[pos]); break;
          case  20 : pos--; tab[pos-1] = TMath::Power(tab[pos-1],tab[pos]); break;
          case  21 : tab[pos-1] = tab[pos-1]*tab[pos-1]; break;
          case  22 : tab[pos-1] = TMath::Sqrt(TMath::Abs(tab[pos-1])); break;
          case  23 : pos2 -= 2; pos++;if (strstr(tab2[pos2],tab2[pos2+1])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  24 : pos--; tab[pos-1] = TMath::Min(tab[pos-1],tab[pos]); break;
          case  25 : pos--; tab[pos-1] = TMath::Max(tab[pos-1],tab[pos]); break;
          case  30 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log(tab[pos-1]);
                     else {tab[pos-1] = 0;} //{indetermination }
                     break;
          case  31 : dexp = tab[pos-1];
                     if (dexp < -700) {tab[pos-1] = 0; break;}
                     if (dexp >  700) {tab[pos-1] = TMath::Exp(700); break;}
                     tab[pos-1] = TMath::Exp(dexp); break;
          case  32 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log10(tab[pos-1]);
                     else {tab[pos-1] = 0;} //{indetermination }
                     break;
          case  40 : pos++; tab[pos-1] = TMath::ACos(-1); break;
          case  41 : tab[pos-1] = TMath::Abs(tab[pos-1]); break;
          case  42 : if (tab[pos-1] < 0) tab[pos-1] = -1; else tab[pos-1] = 1; break;
          case  43 : tab[pos-1] = Double_t(Int_t(tab[pos-1])); break;
          case  50 : pos++; tab[pos-1] = gRandom->Rndm(1); break;
          case  60 : pos--; if (tab[pos-1]!=0 && tab[pos]!=0) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  61 : pos--; if (tab[pos-1]!=0 || tab[pos]!=0) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  62 : pos--; if (tab[pos-1] == tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  63 : pos--; if (tab[pos-1] != tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  64 : pos--; if (tab[pos-1] < tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  65 : pos--; if (tab[pos-1] > tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  66 : pos--; if (tab[pos-1]<=tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  67 : pos--; if (tab[pos-1]>=tab[pos]) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  68 : if (tab[pos-1]!=0) tab[pos-1] = 0; else tab[pos-1] = 1; break;
          case  76 : pos2 -= 2; pos++; if (!strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  77 : pos2 -= 2; pos++;if (strcmp(tab2[pos2+1],tab2[pos2])) tab[pos-1]=1;
                            else tab[pos-1]=0; break;
          case  78 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) & ((Int_t) tab[pos]); break;
          case  79 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) | ((Int_t) tab[pos]); break;
          case  80 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) <<((Int_t) tab[pos]); break;
          case  81 : pos--; tab[pos-1]= ((Int_t) tab[pos-1]) >>((Int_t) tab[pos]); break;
       }
//*-*- Parameter substitution
    } else if (action > 100 && action < 200) {
          pos++;
          tab[pos-1] = fParams[action - 101];
//*-*- Polynomial
    } else if (action > 10000 && action < kConstants) {
          pos++;
          tab[pos-1] = 0;intermede = 1;
          inter2= action/10000; //arrondit
          inter = action/100-100*inter2; //arrondit
          int1=action-inter2*10000-inter*100-1; // aucune simplification ! (sic)
          int2=inter2-1;
          for (j=0 ;j<inter+1;j++) {
              tab[pos-1] += intermede*fParams[j+int1];
              intermede *= x[int2];
          }
//*-*- expo or xexpo or yexpo or zexpo
    } else if (action > 1000 && action < 1500) {
          pos++;
          inter=action/100-10;
          int1=action-inter*100-1000;
          tab[pos-1] = TMath::Exp(fParams[int1-1]+fParams[int1]*x[inter]);
//*-*- xyexpo
    } else if (action > 1500 && action < 1600) {
          pos++;
          int1=action-1499;
          tab[pos-1] = TMath::Exp(fParams[int1-2]+fParams[int1-1]*x[0]+fParams[int1]*x[1]);
//*-*- gaus, xgaus, ygaus or zgaus
    } else if (action > 2000 && action < 2500) {
          pos++;
          inter=action/100-20;
          int1=action-inter*100;
          if (fParams[int1-1999] == 0) {
             intermede2=1e10;
          } else {
             intermede2=Double_t((x[inter]-fParams[int1-2000])/fParams[int1-1999]);
          }
          tab[pos-1] = fParams[int1-2001]*TMath::Exp(-0.5*intermede2*intermede2);
//*-*- xygaus
    } else if (action > 2500 && action < 2600) {
          pos++;
          if (fParams[action-2499] == 0) {
             intermede1=1e10;
          } else {
             intermede1=Double_t((x[0]-fParams[action-2500])/fParams[action-2499]);
          }
          if (fParams[action-2497] == 0) {
             intermede2=1e10;
          } else {
             intermede2=Double_t((x[1]-fParams[action-2498])/fParams[action-2497]);
          }
          tab[pos-1] = fParams[action-2501]*TMath::Exp(-0.5*(intermede1*intermede1+intermede2*intermede2));
//*-*- landau, xlandau, ylandau or zlandau
    } else if (action > 4000 && action < 4500) {
          pos++;
          inter=action/100-40;
          int1=action-inter*100;
          tab[pos-1] = fParams[int1-4001]*TMath::Landau(x[inter],fParams[int1-4000],fParams[int1-3999]);
//*-*- xylandau
    } else if (action > 4500 && action < 4600) {
          pos++;
          intermede1=TMath::Landau(x[0], fParams[action-4500], fParams[action-4499]);
          intermede2=TMath::Landau(x[1], fParams[action-4498], fParams[action-4497]);
          tab[pos-1] = fParams[action-4501]*intermede1*intermede2;
    }
  }
  Double_t result = tab[0];
  return result;
}

//------------------------------------------------------------------------------
TString TFormula::GetExpFormula() const
{
//*-*-*-*-*-*-*-*-*Reconstruct the formula expression from*-*-*-*-*-*-*-*-*-*-*
//*-*              the internal TFormula member variables
//*-*              =======================================
//*-*
//*-*   This function uses the internal member variables of TFormula to
//*-*   construct the mathematical expression associated with the TFormula
//*-*   instance. This function can be used to get an expanded version of the
//*-*   expression originally assigned to the TFormula instance, i.e. that
//*-*   the string returned by GetExpFormula() doesn't depend on other
//*-*   TFormula object names

  if(fNoper>0){
  TString* tab=new TString[fNoper];
  Bool_t* ismulti=new Bool_t[fNoper];
  Int_t spos=0;

  ismulti[0]=kFALSE;
  Int_t optype;
  Int_t j;
  for(Int_t i=0;i<fNoper;i++){
    optype=GetOperType(fOper[i]);
    //Sign inversion
    if(optype==-100){
      tab[spos-1]="-"+tab[spos-1];
      i++;
      continue;
    }
    //Simple name (paramteter,pol0,landau, etc)
    if(optype>=0){
      tab[spos]=fExpr[i];
      ismulti[spos]=kFALSE;
      spos++;
      continue;
    }
    //Basic operators (+,-,*,/,==,^,etc)
    if(optype==-20){
      if(ismulti[spos-2]){
	tab[spos-2]="("+tab[spos-2]+")";
      }
      if(ismulti[spos-1]){
	tab[spos-2]+=fExpr[i]+("("+tab[spos-1]+")");
      }else{
	tab[spos-2]+=fExpr[i]+tab[spos-1];
      }
      ismulti[spos-2]=kTRUE;
      spos--;
      continue;
    }
    //Functions
    if(optype<0){
      tab[spos+optype]=fExpr[i]+("("+tab[spos+optype]);
      for(j=optype+1;j<0;j++){
	tab[spos+optype]+=","+tab[spos+j];
      }
      tab[spos+optype]+=")";
      ismulti[spos+optype]=kFALSE;
      spos+=optype+1;
      continue;
    }
  }
  TString ret=tab[spos-1];
  delete[] tab;
  delete[] ismulti;
  return ret;
  }else{
    TString ret="";
    return ret;
  }
}

//------------------------------------------------------------------------------
Int_t TFormula::GetOperType(Int_t oper) const
{
  if(oper==0) return -100;        //Sign inversion

  if(oper>999 && oper<kConstants) return oper%((int)(oper/100)*100);  //pol0(1), landau(1), ...

  if((oper>0 && oper<6) ||
     oper==20 ||
     (oper>59 && oper<82)) return -20;  //Operators without the format func(x,y,...)

  if((oper>9 && oper<16) ||
     (oper>20 && oper<23) ||
     (oper>29 && oper<34) ||
     (oper>40 && oper<44) ||
     (oper>69 && oper<76)) return -1;  //Functions with the format func(x)

  if(oper>15 && oper<20 ||
     oper>22 && oper<26) return -2;    //Functions with the format func(x,y)

  return 0;

}

//______________________________________________________________________________
Double_t TFormula::GetParameter(Int_t ipar) const
{
  //return value of parameter number ipar

  if (ipar <0 && ipar >= fNpar) return 0;
  return fParams[ipar];
}

//______________________________________________________________________________
Double_t TFormula::GetParameter(const char *parName) const
{
  //return value of parameter named parName

  const Double_t kNaN = 1e-300;
  Int_t index = GetParNumber(parName);
  if (index==-1) {
     Error("TFormula", "Parameter %s not found", parName);
     return kNaN;
  }
  return GetParameter(index);
}

//______________________________________________________________________________
const char *TFormula::GetParName(Int_t ipar) const
{
//*-*-*-*-*-*-*-*Return name of one parameter*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ============================

   if (ipar <0 || ipar >= fNpar) return "";
   if (fNames[ipar].Length() > 0) return (const char*)fNames[ipar];
   return Form("p%d",ipar);
}

//______________________________________________________________________________
Int_t TFormula::GetParNumber(const char *parName) const
{
  // return parameter number by name

   if (!parName)
      return -1;

   for (Int_t i=0; i<fNpar; i++) {
      if (fNames[i] == parName) return i;
   }
  return -1;
}

//______________________________________________________________________________
Bool_t TFormula::IsString(Int_t oper) const
{
   // return true if the expression at the index 'oper' is to be treated as 
   // as string

   return fOper[oper] == kStrings;
}

//______________________________________________________________________________
void TFormula::Print(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*Dump this formula with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  =====================================
   Int_t i;
   Printf(" %20s : %s Ndim= %d, Npar= %d, Noper= %d",GetName(),GetTitle(), fNdim,fNpar,fNoper);
   for (i=0;i<fNoper;i++) {
      Printf(" fExpr[%d] = %s  fOper = %d",i,(const char*)fExpr[i],fOper[i]);
   }
   if (!fNames) return;
   if (!fParams) return;
   for (i=0;i<fNpar;i++) {
      Printf(" Par%3d  %20s = %g",i,GetParName(i),fParams[i]);
   }
}

//______________________________________________________________________________
void TFormula::SetParameter(const char *name, Double_t value)
{
//*-*-*-*-*-*-*-*Initialize parameter number ipar*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ================================

   Int_t ipar = GetParNumber(name);
   if (ipar <0 || ipar >= fNpar) return;
   fParams[ipar] = value;
   Update();
}

//______________________________________________________________________________
void TFormula::SetParameter(Int_t ipar, Double_t value)
{
//*-*-*-*-*-*-*-*Initialize parameter number ipar*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ================================

   if (ipar <0 || ipar >= fNpar) return;
   fParams[ipar] = value;
   Update();
}

//______________________________________________________________________________
void TFormula::SetParameters(const Double_t *params)
{
// Initialize array of all parameters
// see also next function with same name

   for (Int_t i=0; i<fNpar;i++) {
      fParams[i] = params[i];
   }
   Update();
}


//______________________________________________________________________________
void TFormula::SetParameters(Double_t p0,Double_t p1,Double_t p2,Double_t p3,Double_t p4
                       ,Double_t p5,Double_t p6,Double_t p7,Double_t p8,Double_t p9,Double_t p10)
{
// Initialize up to 10 parameters
// All arguments except THE FIRST TWO are optional
// In case of a function with only one parameter, call this function with p1=0.
// Minimum two arguments are required to differentiate this function
// from the SetParameters(cont Double_t *params)

   if (fNpar > 0) fParams[0] = p0;
   if (fNpar > 1) fParams[1] = p1;
   if (fNpar > 2) fParams[2] = p2;
   if (fNpar > 3) fParams[3] = p3;
   if (fNpar > 4) fParams[4] = p4;
   if (fNpar > 5) fParams[5] = p5;
   if (fNpar > 6) fParams[6] = p6;
   if (fNpar > 7) fParams[7] = p7;
   if (fNpar > 8) fParams[8] = p8;
   if (fNpar > 9) fParams[9] = p9;
   if (fNpar >10) fParams[10]= p10;
   Update();
}

//______________________________________________________________________________
void TFormula::SetParName(Int_t ipar, const char *name)
{
// Set name of parameter number ipar

   if (ipar <0 || ipar >= fNpar) return;
   fNames[ipar] = name;
}

//______________________________________________________________________________
void TFormula::SetParNames(const char*name0,const char*name1,const char*name2,const char*name3,const char*name4,
                     const char*name5,const char*name6,const char*name7,const char*name8,const char*name9,const char*name10)
{
//*-*-*-*-*-*-*-*-*-*Set up to 10 parameter names*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ============================

   if (fNpar > 0) fNames[0] = name0;
   if (fNpar > 1) fNames[1] = name1;
   if (fNpar > 2) fNames[2] = name2;
   if (fNpar > 3) fNames[3] = name3;
   if (fNpar > 4) fNames[4] = name4;
   if (fNpar > 5) fNames[5] = name5;
   if (fNpar > 6) fNames[6] = name6;
   if (fNpar > 7) fNames[7] = name7;
   if (fNpar > 8) fNames[8] = name8;
   if (fNpar > 9) fNames[9] = name9;
   if (fNpar >10) fNames[10]= name10;
}

//_______________________________________________________________________
void TFormula::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t v = b.ReadVersion(&R__s, &R__c);
      if (v > 3) {
         TFormula::Class()->ReadBuffer(b, this, v, R__s, R__c);
         if (!TestBit(kNotGlobal)) gROOT->GetListOfFunctions()->Add(this);

         // We need to reinstate (if possible) the TMethodCall.
         if (fFunctions.GetLast()>=0) {
            Compile();
         }
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(b);
      b >> fNdim;
      b >> fNumber;
      if (v > 1) b >> fNval;
      if (v > 2) b >> fNstring;
      fNpar   = b.ReadArray(fParams);
      fNoper  = b.ReadArray(fOper);
      fNconst = b.ReadArray(fConst);
      if (fNoper) {
         fExpr   = new TString[fNoper];
      }
      if (fNpar) {
         fNames  = new TString[fNpar];
      }
      Int_t i;
      for (i=0;i<fNoper;i++)  fExpr[i].Streamer(b);
      for (i=0;i<fNpar;i++)   fNames[i].Streamer(b);
      if (gROOT->GetListOfFunctions()->FindObject(GetName())) return;
      gROOT->GetListOfFunctions()->Add(this);
      b.CheckByteCount(R__s, R__c, TFormula::IsA());
      //====end of old versions

   } else {
      TFormula::Class()->WriteBuffer(b,this);
   }
}
