// @(#)root/hist:$Name:  $:$Id: TFormula.cxx,v 1.4 2000/06/13 10:39:20 brun Exp $
// Author: Nicolas Brun   19/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream.h>
#include <math.h>

#include "TROOT.h"
#include "TFormula.h"
#include "TMath.h"
#include "TRandom.h"

#ifdef WIN32
#pragma optimize("",off)
#endif

static Int_t MAXOP,MAXPAR,MAXCONST;
const Int_t kMAXFOUND = 200;
const Int_t kMAXSTRINGFOUND = 10;
static Int_t already_found[kMAXFOUND];

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
//*-*   Up to 3 dimensions are supported (indicated by x, y, z)
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

  fExpr   = 0;
  fOper   = 0;
  fConst  = 0;
  fParams = 0;
  fNames  = 0;
  fNval   = 0;
  fNstring= 0;

  //eliminate blanks in expression
  Int_t i,j,nch;
  nch = strlen(expression);
  char *expr = new char[nch+1];
  j = 0;
  for (i=0;i<nch;i++) {
     if (expression[i] == ' ') continue;
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
TFormula::TFormula(const TFormula &formula)
{
   ((TFormula&)formula).Copy(*this);
}

//______________________________________________________________________________
TFormula::~TFormula()
{
//*-*-*-*-*-*-*-*-*-*-*Formula default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===========================

   gROOT->GetListOfFunctions()->Remove(this);

   if (fExpr)   { delete [] fExpr;   fExpr   = 0;}
   if (fOper)   { delete [] fOper;   fOper   = 0;}
   if (fConst)  { delete [] fConst;  fConst  = 0;}
   if (fParams) { delete [] fParams; fParams = 0;}
   if (fNames)  { delete [] fNames;  fNames  = 0;}
}


//______________________________________________________________________________
void TFormula::Analyze(const char *schain, Int_t &err)
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
//*-*     %           5
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
//*-*
//*-*   * constants :
//*-*
//*-*    c0  50000      c1  50001  etc..
//*-*
//*-*   * strings :
//*-*
//*-*    s0  80000      s1  80001  etc..
//*-*
//*-*   * variables :
//*-*
//*-*     x    100000     y    100001     z    100002     t    100003
//*-*
//*-*   * parameters :
//*-*
//*-*     [1]        101
//*-*     [2]        102
//*-*     etc.
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
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


   Int_t valeur,find,n,i,j,k,lchain,nomb,virgule,inter;
   Int_t compt,compt2,compt3,compt4,hexa;
   Float_t vafConst;
   UInt_t vafConst2;
   Bool_t parenthese;
   TString s,chaine_error,chaine1ST;
   TString s1,s2,s3,ctemp;
   TString chaine = schain;
   TFormula *oldformula;
   Int_t modulo,plus,puiss10,puiss10bis,moins,multi,divi,puiss,et,ou,petit,grand,egal,diff,peteg,grdeg,etx,oux;
   char t;

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
    lchain = chaine.Length();
    if (lchain==0) err=4;
    else {
      for (i=1; i<=lchain; ++i) {
        if (chaine(i-1,1) == "[") compt2++;
        if (chaine(i-1,1) == "]") compt2--;
        if (chaine(i-1,1) == "(") compt++;
        if (chaine(i-1,1) == ")") compt--;
        if (compt < 0) err = 40; // more open parenthesis than close paraenthesis
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
  modulo=plus=moins=multi=divi=puiss=et=ou=petit=grand=egal=diff=peteg=grdeg=etx=oux=0;

//*-*- Look for simple operators
//*-*  =========================

if (err==0) {
  compt = compt2 = compt3 = compt4 = 0;puiss10=0;puiss10bis = 0;
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
    if (chaine(i-1,1) == "[") compt2++;
    if (chaine(i-1,1) == "]") compt2--;
    if (chaine(i-1,1) == "(") compt++;
    if (chaine(i-1,1) == ")") compt--;
    if (chaine(j-1,1) == "[") compt3++;
    if (chaine(j-1,1) == "]") compt3--;
    if (chaine(j-1,1) == "(") compt4++;
    if (chaine(j-1,1) == ")") compt4--;
    if (chaine(i-1,2)=="&&" && compt==0 && compt2==0 && et==0) {et=i;puiss=0;}
    if (chaine(i-1,2)=="||" && compt==0 && compt2==0 && ou==0) {puiss10=0; ou=i;}
    if (chaine(i-1,1)=="&" && compt==0 && compt2==0 && etx==0) {etx=i;puiss=0;}
    if (chaine(i-1,1)=="|" && compt==0 && compt2==0 && oux==0) {puiss10=0; oux=i;}
    if (chaine(i-1,1)==">" && compt==0 && compt2==0 && grand==0) {puiss10=0; grand=i;}
    if (chaine(i-1,1)=="<" && compt==0 && compt2==0 && petit==0) {puiss10=0; petit=i;}
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
    if (chaine(j-1,1)=="/" && compt4==0 && compt3==0 && divi==0)  {puiss10=0; divi=j;}
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(ou+1,lchain-ou-1);
       Analyze(ctemp.Data(),err);
       fExpr[fNoper] = "||";
       fOper[fNoper] = 61;
       fNoper++;
    }
  } else if (et!=0) {
    if (et==1 || et==lchain-1) {
      err=5;
      chaine_error="&&";
      }
    else {
      ctemp = chaine(0,et-1);
      Analyze(ctemp.Data(),err);
      ctemp = chaine(et+1,lchain-et-1);
      Analyze(ctemp.Data(),err);
      fExpr[fNoper] = "&&";
      fOper[fNoper] = 60;
      fNoper++;
    }
  } else if (oux!=0) {
    if (oux==1 || oux==lchain) {
      err=5;
      chaine_error="|";
      }
    else {
      ctemp = chaine(0,oux-1);
      Analyze(ctemp.Data(),err);
      ctemp = chaine(oux,lchain-oux);
      Analyze(ctemp.Data(),err);
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
      Analyze(ctemp.Data(),err);
      ctemp = chaine(etx,lchain-etx);
      Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(petit,lchain-petit);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(grand,lchain-grand);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(peteg+1,lchain-peteg-1);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(grdeg+1,lchain-grdeg-1);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(egal+1,lchain-egal-1);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(diff+1,lchain-diff-1);
       Analyze(ctemp.Data(),err);
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
       Analyze(ctemp.Data(),err);
       ctemp = chaine(plus,lchain-plus);
       Analyze(ctemp.Data(),err);
       fExpr[fNoper] = "+";
       fOper[fNoper] = 1;
       fNoper++;
    }

  } else {
    if (moins != 0) {
      if (moins == 1) {
        ctemp = chaine(moins,lchain-moins);
        Analyze(ctemp.Data(),err);
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
          Analyze(ctemp.Data(),err);
          ctemp = chaine(moins,lchain-moins);
          Analyze(ctemp.Data(),err);
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
           Analyze(ctemp.Data(),err);
           ctemp = chaine(modulo,lchain-modulo);
           Analyze(ctemp.Data(),err);
           fExpr[fNoper] = "%";
           fOper[fNoper] = 5;
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
          Analyze(ctemp.Data(),err);
          ctemp = chaine(multi,lchain-multi);
          Analyze(ctemp.Data(),err);
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
            Analyze(ctemp.Data(),err);
            ctemp = chaine(divi,lchain-divi);
            Analyze(ctemp.Data(),err);
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
                 Analyze(ctemp.Data(),err);
              } else {
                 ctemp = chaine(0,puiss-1);
                 Analyze(ctemp.Data(),err);
                 ctemp = chaine(puiss,lchain-puiss);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "^";
                 fOper[fNoper] = 20;
                 fNoper++;
              }
            }
//*-*- Look for an already defined expression
          } else {
              find=0;
              oldformula = (TFormula*)gROOT->GetListOfFunctions()->FindObject((const char*)chaine);
               if (oldformula && !oldformula->GetNumber()) {
                 Analyze(oldformula->GetTitle(),err);
                 find=1;
                 if (!err) {
                    Int_t npold = oldformula->GetNpar();
                    fNpar += npold;
                    for (Int_t ipar=0;ipar<npold;ipar++) {
                       fParams[ipar+fNpar-npold] = oldformula->GetParameter(ipar);
                    }
                 }
              }
          if (find == 0) {
//*-*- Check if chaine is a defined variable.
//*-*- Note that DefinedVariable can be overloaded
               k = DefinedVariable(chaine);
               if (k >= 5000 && k < 10000) {
                  fExpr[fNoper] = chaine;
                  fOper[fNoper] = 100000 + k;
                  fNstring++;
                  fNoper++;
               } else if ( k >= 0 ) {
                  fExpr[fNoper] = chaine;
                  fOper[fNoper] = 100000 + k;
                  if (k <kMAXFOUND && !already_found[k]) {
                     already_found[k] = 1;
                     fNval++;
                  }
                  fNoper++;
               } else if (chaine(0,1) == "!") {
                  ctemp = chaine(1,lchain-1);
                  Analyze(ctemp.Data(),err);
                  fExpr[fNoper] = "!";
                  fOper[fNoper] = 68;
                  fNoper++;
               } else if (chaine(0,4) == "cos(") {
                  ctemp = chaine(3,lchain-3);
                  Analyze(ctemp.Data(),err);
                  fExpr[fNoper] = "cos";
                  fOper[fNoper] = 10;
                  fNoper++;
               } else if (chaine(0,4) == "sin(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "sin";
                 fOper[fNoper] = 11;
                 fNoper++;
               } else if (chaine(0,4) == "tan(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "tan";
                 fOper[fNoper] = 12;
                 fNoper++;
               } else if (chaine(0,5) == "acos(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "acos";
                 fOper[fNoper] = 13;
                 fNoper++;
               } else if (chaine(0,5) == "asin(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "asin";
                 fOper[fNoper] = 14;
                 fNoper++;
               } else if (chaine(0,5) == "atan(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "atan";
                 fOper[fNoper] = 15;
                 fNoper++;
                } else if (chaine(0,5) == "cosh(") {
                  ctemp = chaine(4,lchain-4);
                  Analyze(ctemp.Data(),err);
                  fExpr[fNoper] = "cosh";
                  fOper[fNoper] = 70;
                  fNoper++;
               } else if (chaine(0,5) == "sinh(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "sinh";
                 fOper[fNoper] = 71;
                 fNoper++;
               } else if (chaine(0,5) == "tanh(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "tanh";
                 fOper[fNoper] = 72;
                 fNoper++;
               } else if (chaine(0,6) == "acosh(") {
                 ctemp = chaine(5,lchain-5);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "acosh";
                 fOper[fNoper] = 73;
                 fNoper++;
               } else if (chaine(0,6) == "asinh(") {
                 ctemp = chaine(5,lchain-5);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "asinh";
                 fOper[fNoper] = 74;
                 fNoper++;
               } else if (chaine(0,6) == "atanh(") {
                 ctemp = chaine(5,lchain-5);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "atanh";
                 fOper[fNoper] = 75;
                 fNoper++;
               } else if (chaine(0,3) == "sq(") {
                 ctemp = chaine(2,lchain-2);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "sq";
                 fOper[fNoper] = 21;
                 fNoper++;
               } else if (chaine(0,4) == "log(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "log";
                 fOper[fNoper] = 30;
                 fNoper++;
               } else if (chaine(0,6) == "log10(") {
                 ctemp = chaine(5,lchain-5);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "log10";
                 fOper[fNoper] = 32;
                 fNoper++;
               } else if (chaine(0,4) == "exp(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "exp";
                 fOper[fNoper] = 31;
                 fNoper++;
               } else if (chaine(0,4) == "abs(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "abs";
                 fOper[fNoper] = 41;
                 fNoper++;
               } else if (chaine(0,5) == "sign(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "sign";
                 fOper[fNoper] = 42;
                 fNoper++;
               } else if (chaine(0,4) == "int(") {
                 ctemp = chaine(3,lchain-3);
                 Analyze(ctemp.Data(),err);
                 fExpr[fNoper] = "int";
                 fOper[fNoper] = 43;
                 fNoper++;
               } else if (chaine(0,4) == "rndm") {
                 fExpr[fNoper] = "rndm";
                 fOper[fNoper] = 50;
                 fNoper++;
               } else if (chaine(0,5) == "sqrt(") {
                 ctemp = chaine(4,lchain-4);
                 Analyze(ctemp.Data(),err);
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
                   fExpr[fNoper] = "E";
                   fOper[fNoper] = 1001+100*inter2;
                   if (inter2 == 5 && fNpar < 3) fNpar = 3;
                   if (fNpar < 2) fNpar = 2;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      SetNumber(200);
                   }
                }
            } else if (chaine(4,1) == "(") {
                      ctemp = chaine(5,lchain-6);
                      fExpr[fNoper] = "E";
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
                            fOper[fNoper] = 1001+inter+100*inter2;
                            if (inter2 == 5) inter++;
                            if (inter+2>fNpar) fNpar = inter+2;
                            if (fNpar>=MAXPAR) err=7; // too many parameters
                            if (!err) fNoper++;
                            SetNumber(200);
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
                   fExpr[fNoper] = "G";
                   fOper[fNoper] = 2001+100*inter2;
                   if (inter2 == 5 && fNpar < 5) fNpar = 5;
                   if (3>fNpar) fNpar = 3;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      SetNumber(100);
                   }
                }
            } else if (chaine(4,1) == "(" && err==0) {
                      ctemp = chaine(5,lchain-6);
                      fExpr[fNoper] = "G";
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
                             fOper[fNoper] = 2001+inter+100*inter2;
                             if (inter2 == 5) inter += 2;
                             if (inter+3>fNpar) fNpar = inter+3;
                             if (fNpar>=MAXPAR) err=7; // too many parameters
                             if (!err) fNoper++;
                             SetNumber(100);
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
                   fExpr[fNoper] = "L";
                   fOper[fNoper] = 4001+100*inter2;
                   if (inter2 == 5 && fNpar < 5) fNpar = 5;
                   if (3>fNpar) fNpar = 3;
                   if (fNpar>=MAXPAR) err=7; // too many parameters
                   if (!err) {
                      fNoper++;
                      if (fNdim < 1) fNdim = 1;
                      SetNumber(400);
                   }
                }
            } else if (chaine(6,1) == "(" && err==0) {
                      ctemp = chaine(7,lchain-8);
                      fExpr[fNoper] = "L";
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
                             fOper[fNoper] = 4001+inter+100*inter2;
                             if (inter2 == 5) inter += 2;
                             if (inter+3>fNpar) fNpar = inter+3;
                             if (fNpar>=MAXPAR) err=7; // too many parameters
                             if (!err) fNoper++;
                             SetNumber(400);
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
              fExpr[fNoper] = "P";
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
            compt = 4; nomb = 0; virgule = 0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == ",") nomb++;
              if (nomb == 1 && virgule == 0) virgule = compt;
            }
            if (nomb != 1) err = 22; // There are plus or minus than 2 arguments for pow
            else {
              ctemp = chaine(4,virgule-5);
              Analyze(ctemp.Data(),err);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err);
              fExpr[fNoper] = "";
              fOper[fNoper] = 20;
              fNoper++;
            }
		  } else if (chaine(0,7) == "strstr(") {
            compt = 7; nomb = 0; virgule = 0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == ",") nomb++;
              if (nomb == 1 && virgule == 0) virgule = compt;
            }
            if (nomb != 1) err = 28; // There are plus or minus than 2 arguments for strstr
            else {
              ctemp = chaine(7,virgule-8);
              Analyze(ctemp.Data(),err);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err);
              fExpr[fNoper] = "";
              fOper[fNoper] = 23;
              fNoper++;
            }

          } else if (chaine(0,6) == "atan2(") {
            compt = 6; nomb = 0; virgule = 0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == ",") nomb++;
              if (nomb == 1 && virgule == 0) virgule = compt;
            }
            if (nomb != 1) err = 21;  //{ There are plus or minus than 2 arguments for atan2
            else {
              ctemp = chaine(6,virgule-7);
              Analyze(ctemp.Data(),err);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err);
              fExpr[fNoper] = "";
              fOper[fNoper] = 16;
              fNoper++;
            }
          } else if (chaine(0,5) == "fmod(") {
            compt = 5; nomb = 0; virgule = 0;
            while(compt != lchain) {
              compt++;
              if (chaine(compt-1,1) == ",") nomb++;
              if (nomb == 1 && virgule == 0) virgule = compt;
            }
            if (nomb != 1) err = 21;  //{ There are plus or minus than 2 arguments for atan2
            else {
              ctemp = chaine(5,virgule-6);
              Analyze(ctemp.Data(),err);
              ctemp = chaine(virgule,lchain-virgule-1);
              Analyze(ctemp.Data(),err);
              fExpr[fNoper] = "";
              fOper[fNoper] = 17;
              fNoper++;
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
              fOper[fNoper-1] = valeur + 101;
            }
          } else if (chaine == "pi") {
            fExpr[fNoper] = "P";
            fOper[fNoper] = 40;
            fNoper++;
//*-*- None of the above. Must be a numerical expression
//*-*  =================================================
          }
//*-*- Maybe it is a string
          else
            if (chaine(0,1)=="\"" && chaine(chaine.Length()-1,1)=="\"") {
               //*-* It is a string !!!
               fExpr[fNoper] = chaine(1,chaine.Length()-2);
               fOper[fNoper] = 80000;
               fNoper++;
            }
            else {
            compt=0;compt2=0;hexa=0;
            if ((chaine(0,2)=="0x")||(chaine(0,2)=="0X")) hexa=1;
            for (j=0; j<chaine.Length(); j++) {
                t=chaine[j];
                if (hexa==0) {
              	      if (chaine(j,1)=="e" || chaine(j,2)=="e+" || chaine(j,2)=="e-") {
                 	  compt2++;
                 	  compt = 0;
                 	  if (chaine(j,2)=="e+" || chaine(j,2)=="e-") j++;
                	  if (compt2>1) {
                	       err=26;
                 	       chaine_error=chaine;
                	  }
                      }
                      else {
                          if (chaine(j,1) == ".") compt++;
                          else {
                             if (!strchr("0123456789",t) && (chaine(j,1)!="+" || j!=0) && compt!=1) {
                                err=26;
                                chaine_error=chaine;
                             }
                          }
                      }
                }
                else {
                      if (!strchr("0123456789abcdefABCDEF",t) && (j>1)) {
                          err=26;
                          chaine_error=chaine;
                      }
                }
            }
            if (fNconst >= MAXCONST) err = 27;
            if (!err) {
               if (hexa==0) {if (sscanf((const char*)chaine,"%g",&vafConst) > 0) err = 0; else err =1;}
               else {if (sscanf((const char*)chaine,"%x",&vafConst2) > 0) err = 0; else err=1;
               vafConst = (Float_t) vafConst2;}
               fExpr[fNoper] = chaine;
               k = -1;
               for (j=0;j<fNconst;j++) {
                  if (vafConst == fConst[j] ) k= j;
               }
               if ( k < 0) {  k = fNconst; fNconst++; fConst[k] = vafConst; }
               fOper[fNoper] = 50000 + k;
               fNoper++;
            }
          }
        }
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
      case 40 : cout<<" '(' is expected"<<endl; break;
      case 41 : cout<<" ')' is expected"<<endl; break;
      case 42 : cout<<" '[' is expected"<<endl; break;
      case 43 : cout<<" ']' is expected"<<endl; break;
     }
  err=1;
  }

}

//______________________________________________________________________________
Int_t TFormula::Compile(const char *expression)
{
//*-*-*-*-*-*-*-*-*-*-*Compile expression already stored in fTitle*-*-*-*-*-*
//*-*                  ===========================================
//*-*
//*-*   Loop on all subexpressions of formula stored in fTitle
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


//*-*- If expression is not empty, take it, otherwise take the title
  if (strlen(expression)) SetTitle(expression);

  TString chaine = GetTitle();
//  chaine.ToLower();

  MAXOP   = 1000;
  MAXPAR  = 100;
  MAXCONST= 100;
  if (fExpr)   { delete [] fExpr;   fExpr   = 0;}
  if (fOper)   { delete [] fOper;   fOper   = 0;}
  if (fConst)  { delete [] fConst;  fConst  = 0;}
  if (fParams) { delete [] fParams; fParams = 0;}
  if (fNames)  { delete [] fNames;  fNames  = 0;}
  fNpar   = 0;
  fNdim   = 0;
  fNoper  = 0;
  fNconst = 0;
  fNstring= 0;
  fNumber = 0;
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
  for(i=0;i<kMAXFOUND;i++) already_found[i]=0;
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
        SetParName(1,"Mean");
        SetParName(2,"Sigma");
     }
  }


//*-* replace 'normal' == or != by ==(string) or !=(string) if needed.
  Int_t is_it_string,last_string=0;
  if (!fOper) fNoper = 0;
  for (i=0; i<fNoper; i++) {
     is_it_string = 0;
     if ((fOper[i]>=105000 && fOper[i]<110000) || fOper[i] == 80000) is_it_string = 1;
     else if (fOper[i] == 62 && last_string == 1) fOper[i] = 76;
     else if (fOper[i] == 63 && last_string == 1) fOper[i] = 77;
     last_string = is_it_string;
  }

  if (err) { fNdim = 0; return 1; }
  return 0;
}

//______________________________________________________________________________
void TFormula::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =================

   Int_t i;
   TNamed::Copy(obj);
   ((TFormula&)obj).fNdim   = fNdim;
   ((TFormula&)obj).fNpar   = fNpar;
   ((TFormula&)obj).fNoper  = fNoper;
   ((TFormula&)obj).fNconst = fNconst;
   ((TFormula&)obj).fNumber = fNumber;
   ((TFormula&)obj).fNval   = fNval;
   if (fNoper)  ((TFormula&)obj).fExpr   = new TString[fNoper];
   if (fNoper)  ((TFormula&)obj).fOper   = new Int_t[fNoper];
   if (fNconst) ((TFormula&)obj).fConst  = new Double_t[fNconst];
   if (fNpar)   ((TFormula&)obj).fParams = new Double_t[fNpar];
   if (fNpar)   ((TFormula&)obj).fNames  = new TString[fNpar];
   for (i=0;i<fNoper;i++)  ((TFormula&)obj).fExpr[i]   = fExpr[i];
   for (i=0;i<fNoper;i++)  ((TFormula&)obj).fOper[i]   = fOper[i];
   for (i=0;i<fNconst;i++) ((TFormula&)obj).fConst[i]  = fConst[i];
   for (i=0;i<fNpar;i++)   ((TFormula&)obj).fParams[i] = fParams[i];
   for (i=0;i<fNpar;i++)   ((TFormula&)obj).fNames[i]  = fNames[i];
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
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
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
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
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
Double_t TFormula::Eval(Double_t x, Double_t y, Double_t z)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
//*-*
//*-*   The current value of variables x,y,z is passed through x, y and z.
//*-*   The parameters used will be the ones in the array params if params is given
//*-*    otherwise parameters will be taken from the stored data members fParams
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  Double_t xx[3];
  xx[0] = x;
  xx[1] = y;
  xx[2] = z;
  return EvalPar(xx);

}

//______________________________________________________________________________
Double_t TFormula::EvalPar(Double_t *x, Double_t *params)
{
//*-*-*-*-*-*-*-*-*-*-*Evaluate this formula*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =====================
//*-*
//*-*   The current value of variables x,y,z is passed through the pointer x.
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
//*-*- a variable
    if (action >= 110000) {
       pos++; tab[pos-1] = x[action-110000];
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
    if (action >= 100000) {
       if (!precalculated) {
          precalculated = 1;
          for(j=0;j<fNval;j++) param_calc[j]=DefinedValue(j);
       }
       pos++; tab[pos-1] = param_calc[action-100000];
       continue;
    }
//*-*- String
    if (action == 80000) {
       pos2++; tab2[pos2-1] = (char*)fExpr[i].Data();
       continue;
    }
//*-*- numerical value
    if (action >= 50000) {
       pos++; tab[pos-1] = fConst[action-50000];
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
          case  30 : if (tab[pos-1] > 0) tab[pos-1] = TMath::Log(tab[pos-1]);
                     else {tab[pos-1] = 0;} //{indetermination }
                     break;
          case  31 : dexp = tab[pos-1];
                     if (dexp < -70) {tab[pos-1] = 0; break;}
                     if (dexp >  70) {tab[pos-1] = TMath::Exp(70); break;}
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
       }
//*-*- Parameter substitution
    } else if (action > 100 && action < 200) {
          pos++;
          tab[pos-1] = fParams[action - 101];
//*-*- Polynomial
    } else if (action > 10000 && action < 50000) {
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
          intermede2=Double_t((x[inter]-fParams[int1-2000])/fParams[int1-1999]);
          tab[pos-1] = fParams[int1-2001]*TMath::Exp(-0.5*intermede2*intermede2);
//*-*- xygaus
    } else if (action > 2500 && action < 2600) {
          pos++;
          intermede1=Double_t((x[0]-fParams[action-2500])/fParams[action-2499]);
          intermede2=Double_t((x[1]-fParams[action-2498])/fParams[action-2497]);
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

//______________________________________________________________________________
Double_t TFormula::GetParameter(const char *parName) 
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

   if (ipar <0 && ipar >= fNpar) return "";
   if (fNames[ipar].Length() > 0) return (const char*)fNames[ipar];
   return Form("p%d",ipar);
}

//______________________________________________________________________________
Int_t TFormula::GetParNumber(const char *parName) 
{
  // return parameter number by name
   
   for (Int_t i=0; i<MAXPAR; i++) {
      if (fNames[i] == parName) return i;
   }
  return -1;
}

//______________________________________________________________________________
void TFormula::Print(Option_t *)
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
void TFormula::SetParameter(Int_t ipar, Double_t value)
{
//*-*-*-*-*-*-*-*Initialize parameter number ipar*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ================================

   if (ipar <0 || ipar >= fNpar) return;
   fParams[ipar] = value;
   Update();
}

//______________________________________________________________________________
void TFormula::SetParameters(Double_t *params)
{
//*-*-*-*-*-*-*-*Initialize array of all parameters*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ==================================

   for (Int_t i=0; i<fNpar;i++) {
      fParams[i] = params[i];
   }
   Update();
}


//______________________________________________________________________________
void TFormula::SetParameters(Double_t p0,Double_t p1,Double_t p2,Double_t p3,Double_t p4
                       ,Double_t p5,Double_t p6,Double_t p7,Double_t p8,Double_t p9)
{
//*-*-*-*-*-*-*-*Initialize up to 10 parameters*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*            ==============================

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
   Update();
}

//______________________________________________________________________________
void TFormula::SetParNames(const char*name0,const char*name1,const char*name2,const char*name3,const char*name4,
                     const char*name5,const char*name6,const char*name7,const char*name8,const char*name9)
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
}

//_______________________________________________________________________
void TFormula::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   Int_t i;
   UInt_t R__s, R__c;
   if (b.IsReading()) {
      Version_t v = b.ReadVersion(&R__s, &R__c);
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
      for (i=0;i<fNoper;i++)  fExpr[i].Streamer(b);
      for (i=0;i<fNpar;i++)   fNames[i].Streamer(b);
      if (gROOT->GetListOfFunctions()->FindObject(GetName())) return;
      gROOT->GetListOfFunctions()->Add(this);
      b.CheckByteCount(R__s, R__c, TFormula::IsA());
   } else {
      R__c = b.WriteVersion(TFormula::IsA(), kTRUE);
      TNamed::Streamer(b);
      b << fNdim;
      b << fNumber;
      b << fNval;
      b << fNstring;
      b.WriteArray(fParams,fNpar);
      b.WriteArray(fOper,fNoper);
      b.WriteArray(fConst,fNconst);
      for (i=0;i<fNoper;i++)  fExpr[i].Streamer(b);
      for (i=0;i<fNpar;i++)  fNames[i].Streamer(b);
      b.SetByteCount(R__c, kTRUE);
   }
}
