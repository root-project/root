// @(#)root/hist:$Id$
// Author: Maciej Zimnoch 30/09/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#if __cplusplus >= 201103L
#define ROOT_CPLUSPLUS11 1
#endif

#include "TROOT.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMath.h"
#include "TMethodCall.h"
#include <TBenchmark.h>
#include "TError.h"
#include "TInterpreter.h"
#include "TFormula.h"


#ifdef WIN32
#pragma optimize("",off)
#endif

ClassImp(TFormula)
//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*The  F O R M U L A  class*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =========================
//*-*
//*-*   This class has been implemented during Google Summer of Code 2013 by Maciej Zimnoch.
//*-*   =========================================================
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
//*-*     -  gaus(0)*expo(3)  + ypol3(5)*x
//*-*     -  gausn(0)*expo(3) + ypol3(5)*x
//*-*
//*-*  In the last example above:
//*-*     gaus(0) is a substitute for [0]*exp(-0.5*((x-[1])/[2])**2)
//*-*        and (0) means start numbering parameters at 0
//*-*     gausn(0) is a substitute for [0]*exp(-0.5*((x-[1])/[2])**2)/(sqrt(2*pi)*[2]))
//*-*        and (0) means start numbering parameters at 0
//*-*     expo(3) is a substitute for exp([3]+[4]*x)
//*-*     pol3(5) is a substitute for par[5]+par[6]*x+par[7]*x**2+par[8]*x**3
//*-*         (PolN stands for Polynomial of degree N)
//*-*
//*-*   TMath functions can be part of the expression, eg:
//*-*     -  TMath::Landau(x)*sin(x)
//*-*     -  TMath::Erf(x)
//*-*
//*-*   Formula may contain constans, eg:
//*-*    - sqrt2 
//*-*    - e  
//*-*    - pi 
//*-*    - ln10 
//*-*    - infinity
//*-*      and more.
//*-*   
//*-*   Comparisons operators are also supported (&&, ||, ==, <=, >=, !)
//*-*   Examples:
//*-*      sin(x*(x<0.5 || x>1))
//*-*   If the result of a comparison is TRUE, the result is 1, otherwise 0.
//*-*
//*-*   Already predefined names can be given. For example, if the formula
//*-*     TFormula old("old",sin(x*(x<0.5 || x>1))) one can assign a name to the formula. By default
//*-*     the name of the object = title = formula itself.
//*-*     TFormula new("new","x*old") is equivalent to:
//*-*     TFormula new("new","x*sin(x*(x<0.5 || x>1))")
//*-*
//*-*   Class supports unlimited numer of variables and parameters.
//*-*   By default it has 4 variables(indicated by x,y,z,t) and no parameters.
//*-*
//*-*   This class is the base class for the function classes TF1,TF2 and TF3.
//*-*   It is also used by the ntuple selection mechanism TNtupleFormula.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Bool_t TFormula::IsOperator(const char c)
{
   char ops[] = { '+','^', '-','/','*','<','>','|','&','!','='};
   Int_t opsLen = sizeof(ops)/sizeof(char);
   for(Int_t i = 0; i < opsLen; ++i)
      if(ops[i] == c)
         return true;
   return false;
}

Bool_t TFormula::IsBracket(const char c)
{
   char brackets[] = { ')','(','{','}'};
   Int_t bracketsLen = sizeof(brackets)/sizeof(char);
   for(Int_t i = 0; i < bracketsLen; ++i)
      if(brackets[i] == c)
         return true;
   return false;
}

Bool_t TFormula::IsFunctionNameChar(const char c)
{
   return !IsBracket(c) && !IsOperator(c) && c != ',';
}

Bool_t TFormula::IsDefaultVariableName(const TString &name)
{
   return name == "x" || name == "z" || name == "y" || name == "t";
}

TFormula::TFormula()
{
   fName = "";
   fTitle = "";
   fClingInput = "";
   fReadyToExecute = false;
   fClingInitialized = false;
   fAllParametersSetted = false;
   fMethod = 0;
   fNdim = 0;
   fNpar = 0;
   fNumber = 0;
   fNamePrefix = "";
   fFormula = "";
}

TFormula::~TFormula()
{
   if(fMethod)
   {  
      fMethod->Delete();
   }
}

TFormula::TFormula(const char *name, Int_t nparams, Int_t ndims)
{
   //*-*     
   //*-*  Constructor
   //*-*  When TF1 is constructed using C++ function, TF1 need space to keep parameters values.
   //*-*  
   fName = name;
   fTitle = "";
   fClingInput = "";
   fReadyToExecute = false;
   fClingInitialized = false;
   fAllParametersSetted = false;
   fMethod = 0;
   fNdim = ndims;
   fNpar = 0;
   fNumber = 0;
   fNamePrefix = "";
   fFormula = "";
   FillDefaults();
   for(Int_t i = 0; i < nparams; ++i)
   {
      TString parName = TString::Format("%d",i);
      AddParameter(parName,0);
   }
}

TFormula::TFormula(const TString &name, TString formula)
   :fClingInput(formula),fFormula(formula)
{
   fReadyToExecute = false;
   fClingInitialized = false;
   fMethod = 0;
   fNamePrefix = "TF__";
   fName = fNamePrefix + name;
   TNamed(fName,formula);
   fNdim = 0;
   fNpar = 0;
   fNumber = 0;
   FillDefaults();

   TFormula *old = (TFormula*)gROOT->GetListOfFunctions()->FindObject(fName);
   if (old) 
   {
      gROOT->GetListOfFunctions()->Remove(old);
   }
   if (name == "x" || name == "y" || name == "z" || name == "t")
   {
      Error("TFormula","The name %s is reserved as a TFormula variable name.\n",name.Data());
   } else 
   {

      gROOT->GetListOfFunctions()->Add(this);

   }
   PreProcessFormula(fFormula);
   fClingInput = fFormula;
   PrepareFormula(fClingInput);

}

TFormula::TFormula(const TFormula &formula) : TNamed(formula.GetName(),formula.GetTitle())
{
   fReadyToExecute = false;
   fClingInitialized = false;
   fMethod = 0;
   fNamePrefix = "TF__";
   fName = fNamePrefix + formula.GetName();
   fNdim = formula.GetNdim();
   fNpar = formula.GetNpar();
   fNumber = formula.GetNumber();
   fFormula = formula.GetExpFormula();
   FillDefaults();

   TFormula *old = (TFormula*)gROOT->GetListOfFunctions()->FindObject(formula.GetName());
   if (old) 
   {
      gROOT->GetListOfFunctions()->Remove(old);
   }
   if (strcmp(formula.GetName(),"x") == 0 || strcmp(formula.GetName(),"y") == 0 ||
       strcmp(formula.GetName(),"z") == 0 || strcmp(formula.GetName(),"t") == 0)
   {
      Error("TFormula","The name %s is reserved as a TFormula variable name.\n",formula.GetName());
   } else 
   {
      gROOT->GetListOfFunctions()->Add(this);
   }
   PreProcessFormula(fFormula);
   fClingInput = fFormula;
   PrepareFormula(fClingInput);
}

TFormula& TFormula::operator=(const TFormula &rhs)
{
   //*-*    
   //*-*  = Operator    
   //*-*    

   if (this != &rhs) {
      rhs.Copy(*this);
   }
   return *this;
}
void TFormula::Copy(TObject &obj) const
{
   ((TFormula&)obj).fFuncs = fFuncs;
   ((TFormula&)obj).fVars = fVars;
   ((TFormula&)obj).fParams = fParams;
   ((TFormula&)obj).fConsts = fConsts;
   ((TFormula&)obj).fFunctionsShortcuts = fFunctionsShortcuts;
   ((TFormula&)obj).fFormula  = fFormula;
   ((TFormula&)obj).fNdim = fNdim;
   ((TFormula&)obj).fNpar = fNpar;
   ((TFormula&)obj).fNumber = fNumber;
   ((TFormula&)obj).fLinearParts = fLinearParts;
   ((TFormula&)obj).SetParameters(GetParameters());
}
void TFormula::PrepareEvalMethod()
{
   //*-*    
   //*-*    Sets TMethodCall to function inside Cling environment
   //*-*    TFormula uses it to execute function.
   //*-*    After call, TFormula should be ready to evaluate formula.   
   //*-*    
   if(!fMethod)
   {
      fMethod = new TMethodCall();

      Bool_t hasParameters = (fNpar > 0);
      Bool_t hasVariables  = (fNdim > 0);
      TString prototypeArguments = "";
      if(hasVariables)
      {
         prototypeArguments.Append("Double_t*");
      }
      if(hasVariables && hasParameters)
      {
         prototypeArguments.Append(",");
      }
      if(hasParameters)
      {
         prototypeArguments.Append("Double_t*");
      }
      fMethod->InitWithPrototype(fName,prototypeArguments);
      if(!fMethod->IsValid())
      {
         Error("Eval","Can't find %s function prototype with arguments %s",fName.Data(),prototypeArguments.Data());
         return ;
      }
      if(hasParameters)
      {
         Long_t args[2];
         args[0] = (Long_t)fClingVariables.data();
         args[1] = (Long_t)fClingParameters.data();
         fMethod->SetParamPtrs(args,2);
      }
      else
      {
         Long_t args[1];
         args[0] = (Long_t)fClingVariables.data();
         fMethod->SetParamPtrs(args,1);
      }
   }
}

void TFormula::InputFormulaIntoCling()
{
   //*-*    
   //*-*    Inputs formula, transfered to C++ code into Cling
   //*-*    
   if(!fClingInitialized && fReadyToExecute && fClingInput.Length() > 0)
   {
      char rawInputOn[]  = ".rawInput 1";
      char rawInputOff[] = ".rawInput 0";
      gCling->ProcessLine(rawInputOn);
      gCling->ProcessLine(fClingInput);
      gCling->ProcessLine(rawInputOff);
      PrepareEvalMethod();
      fClingInitialized = true;
   }
}
void TFormula::FillDefaults()
{
   //*-*    
   //*-*    Fill structures with default variables, constants and function shortcuts
   //*-*    
#ifdef ROOT_CPLUSPLUS11
   const TString defvars[] = { "x","y","z","t"};
   const pair<TString,Double_t> defconsts[] = { {"pi",TMath::Pi()}, {"sqrt2",TMath::Sqrt2()},
         {"infinity",TMath::Infinity()}, {"e",TMath::E()}, {"ln10",TMath::Ln10()},
         {"loge",TMath::LogE()}, {"c",TMath::C()}, {"g",TMath::G()}, 
         {"h",TMath::H()}, {"k",TMath::K()},{"sigma",TMath::Sigma()},
         {"r",TMath::R()}, {"eg",TMath::EulerGamma()},{"true",1},{"false",0} }; 
   const pair<TString,TString> funShortcuts[] = { {"sin","TMath::Sin" },
         {"cos","TMath::Cos" }, {"exp","TMath::Exp"}, {"log","TMath::Log"},
         {"tan","TMath::Tan"}, {"sinh","TMath::SinH"}, {"cosh","TMath::CosH"},
         {"tanh","TMath::TanH"}, {"asin","TMath::ASin"}, {"acos","TMath::ACos"},
         {"atan","TMath::ATan"}, {"atan2","TMath::ATan2"}, {"sqrt","TMath::Sqrt"},
         {"ceil","TMath::Ceil"}, {"floor","TMath::Floor"}, {"pow","TMath::Power"},
         {"binomial","TMath::Binomial"},{"abs","TMath::Abs"} }; 
   for(auto var : defvars)
   {
      fVars[var] = TFormulaVariable(var,0,fVars.size());
      fClingVariables.push_back(0);
   }
   for(auto con : defconsts)
   {
      fConsts[con.first] = con.second;
   }
   for(auto fun : funShortcuts)
   {
      fFunctionsShortcuts[fun.first] = fun.second;
   }
#else
   TString  defvarsNames[] = {"x","y","z","t"};
   Int_t    defvarsLength = sizeof(defvarsNames)/sizeof(TString);

   TString  defconstsNames[] = {"pi","sqrt2","infinity","e","ln10","loge","c","g","h","k","sigma","r","eg","true","false"};
   Double_t defconstsValues[] = {TMath::Pi(),TMath::Sqrt2(),TMath::Infinity(),TMath::E(),TMath::Ln10(),TMath::LogE(),
                                 TMath::C(),TMath::G(),TMath::H(),TMath::K(),TMath::Sigma(),TMath::R(),TMath::EulerGamma()};
   Int_t    defconstsLength = sizeof(defconstsNames)/sizeof(TString);

   TString  funShortcutsNames[] = {"sin","cos","exp","log","tan","sinh","cosh","tanh","asin","acos","atan","atan2","sqrt",
                                  "ceil","floor","pow","binomial","abs"};
   TString  funShortcutsExtendedNames[] = {"TMath::Sin","TMath::Cos","TMath::Exp","TMath::Log","TMath::Tan","TMath::SinH",
                                           "TMath::CosH","TMath::TanH","TMath::ASin","TMath::ACos","TMath::ATan","TMath::ATan2",
                                           "TMath::Sqrt","TMath::Ceil","TMath::Floor","TMath::Power","TMath::Binomial","TMath::Abs"};
   Int_t    funShortcutsLength = sizeof(funShortcutsNames)/sizeof(TString);                                           

   for(Int_t i = 0; i < defvarsLength; ++i)
   {
      TString var = defvarsNames[i];
      Double_t value = 0;
      fVars[var] = TFormulaVariable(var,value,fVars.size());
      fClingVariables.push_back(value);
   }

   for(Int_t i = 0; i < defconstsLength; ++i)
   {
      fConsts[defconstsNames[i]] = defconstsValues[i];
   }
   for(Int_t i = 0; i < funShortcutsLength; ++i)
   {
      pair<TString,TString> fun(funShortcutsNames[i],funShortcutsExtendedNames[i]);
      fFunctionsShortcuts[fun.first] = fun.second;
   }

#endif 

}

void TFormula::HandlePolN(TString &formula)
{
   //*-*    
   //*-*    Handling polN
   //*-*    If before 'pol' exist any name, this name will be treated as variable used in polynomial
   //*-*    eg.
   //*-*    varpol2(5) will be replaced with: [5] + [6]*var + [7]*var^2
   //*-*    Empty name is treated like variable x.
   //*-*    
   Int_t polPos = formula.Index("pol");
   while(polPos != kNPOS)
   {
      SetBit(kLinear,1);

      Bool_t defaultVariable = false;
      TString variable;
      Int_t openingBracketPos = formula.Index('(',polPos);
      Bool_t defaultCounter = (openingBracketPos == kNPOS);
      Bool_t defaultDegree = true;
      Int_t degree,counter;
      if(!defaultCounter)
      {
          degree = TString(formula(polPos + 3,openingBracketPos - polPos - 3)).Atoi();
          counter = TString(formula(openingBracketPos+1,formula.Index(')',polPos) - openingBracketPos)).Atoi(); 
      }
      else
      {
         Int_t temp = polPos+3;
         while(temp < formula.Length() && isdigit(formula[temp]))
         {
            defaultDegree = false;
            temp++;
         }
         degree = TString(formula(polPos+3,temp - polPos - 3)).Atoi();
         counter = 0;
      }
      fNumber = 300 + degree;
      TString replacement = TString::Format("[%d]",counter);
      if(polPos - 1 < 0 || !IsFunctionNameChar(formula[polPos-1]))
      {
         variable = "x";
         defaultVariable = true;
      }
      else
      {
         Int_t tmp = polPos - 1;
         while(tmp >= 0 && IsFunctionNameChar(formula[tmp]))
         {
            tmp--;
         }
         variable = formula(tmp + 1, polPos - (tmp+1));
      }
      Int_t param = counter + 1;
      Int_t tmp = 1;
      while(tmp <= degree)
      {
         replacement.Append(TString::Format("+[%d]*%s^%d",param,variable.Data(),tmp));
         param++;
         tmp++;
      }
      TString pattern;
      if(defaultCounter && !defaultDegree)
      {
         pattern = TString::Format("%spol%d",(defaultVariable ? "" : variable.Data()),degree);
      }
      else if(defaultCounter && defaultDegree)
      {
         pattern = TString::Format("%spol",(defaultVariable ? "" : variable.Data()));
      }
      else
      {
         pattern = TString::Format("%spol%d(%d)",(defaultVariable ? "" : variable.Data()),degree,counter);   
      }
      
      formula.ReplaceAll(pattern,replacement);
      polPos = formula.Index("pol");
   }
}
void TFormula::HandleParametrizedFunctions(TString &formula)
{
   //*-*    
   //*-*    Handling parametrized functions
   //*-*    Function can be normalized, and have different variable then x.
   //*-*    Name before function will be used as variable inside.
   //*-*    Empty name is treated like varibale x.
   //*-*    Normalized function has char 'n' after name, eg.
   //*-*    vargausn(0) will be replaced with [0]*exp(-0.5*((var-[1])/[2])^2)/(sqrt(2*pi)*[2])
   //*-*    
   //*-*    Adding function is easy, just follow these rules:
   //*-*    - first tuple argument is name of function 
   //*-*    - second tuple argument is function body
   //*-*    - third tuple argument is normalized function body 
   //*-*    - {V} is a place where variable appear
   //*-*    - [num] stands for parameter number. 
   //*-*      If user pass to function argument 5, num will stand for (5 + num) parameter.
   //*-*    
   map<TString,pair<TString,TString> > functions;
   functions["gaus"] = pair<TString,TString>("[0]*exp(-0.5*(({V}-[1])/[2])^2)","[0]*exp(-0.5*(({V}-[1])/[2])^2)/(sqrt(2*pi)*[2])");
   functions["landau"] = pair<TString,TString>("TMath::Landau({V},[0],[1],false)","TMath::Landau({V},[0],[1],true)");
   functions["expo"] = pair<TString,TString>("exp([0]+[1]*{V})","");
   map<TString,Int_t> functionsNumbers;
   functionsNumbers["gaus"] = 100;
   functionsNumbers["landau"] = 200;
   functionsNumbers["expo"] = 400;
   for(map<TString,pair<TString,TString> >::iterator it = functions.begin(); it != functions.end(); ++it) 
   {

      TString funName = it->first;
      Int_t funPos = formula.Index(funName);
      while(funPos != kNPOS)
      {
         fNumber = functionsNumbers[funName];
         Bool_t isNormalized = (formula[funPos + funName.Length()] == 'n');
         TString body = (isNormalized ? it->second.second : it->second.first);
         if(isNormalized && body == "")
         {
            Error("PreprocessFormula","Function %s has no normalized form.",funName.Data());
            break;
         }
         if(isNormalized)
         {
            SetBit(kNormalized,1);
         }
         TString variable;
         Bool_t defaultVariable = false;
         if(funPos - 1 < 0 || !IsFunctionNameChar(formula[funPos-1]))
         {
            variable = "x";
            defaultVariable = true;
         }
         else
         {
            Int_t tmp = funPos - 1;
            while(tmp >= 0 && IsFunctionNameChar(formula[tmp]))
            {
               tmp--;
            }
            variable = formula(tmp + 1, funPos - (tmp+1));

         }
         Int_t openingBracketPos = formula.Index('(',funPos);
         Bool_t defaultCounter = (openingBracketPos == kNPOS);
         Int_t counter;
         if(defaultCounter)
         {
            counter = 0;           
         }
         else
         {
            counter = TString(formula(openingBracketPos+1,formula.Index(')',funPos) - openingBracketPos -1)).Atoi(); 
         }
         for(int i = 0 ; i < body.Length() ; ++i)
         {
            if(body[i] == '{')
            {
               body.Replace(i,3,variable,variable.Length());
               i += variable.Length();
            }
            else if(body[i] == '[')
            {
               Int_t tmp = i;
               while(tmp < body.Length() && body[tmp] != ']')
               {
                  tmp++;
               }
               Int_t num = TString(body(i+1,tmp - 1 - i)).Atoi();
               num += counter;
               TString replacement = TString::Format("%d",num);
              
               body.Replace(i+1,tmp - 1 - i,replacement,replacement.Length());
               i += replacement.Length() + 1;
            }
         }
         TString pattern;
         if(defaultCounter)
         {
            pattern = TString::Format("%s%s%s",
                           (defaultVariable ? "" : variable.Data()),
                           funName.Data(),
                           (isNormalized ? "n" : ""));
         }
            pattern = TString::Format("%s%s%s(%d)",
                           (defaultVariable ? "" : variable.Data()),
                           funName.Data(),
                           (isNormalized ? "n" : ""),
                           counter);
         TString replacement = body;
         if(!defaultVariable)
         {
            funPos -= variable.Length();
         }
         formula.Replace(funPos,pattern.Length(),replacement,replacement.Length());

         funPos = formula.Index(funName);
      }
   }
}
void TFormula::HandleExponentiation(TString &formula)
{
   //*-*    
   //*-*    Handling exponentiation
   //*-*    Can handle multiple carets, eg.
   //*-*    2^3^4 will be treated like 2^(3^4)    
   //*-*    
   Int_t caretPos = formula.Last('^');
   while(caretPos != kNPOS)
   {

      TString right,left;
      Int_t temp = caretPos;
      temp--;
      if(formula[temp] == ')')
      {
         Int_t depth = 1;
         temp--;
         while(depth != 0 && temp > 0)
         {
            if(formula[temp] == ')')
               depth++;
            if(formula[temp] == '(')
               depth--;
            temp--;
         }
         temp++;
      }
      while(temp >= 0 && !IsOperator(formula[temp]))
      {
         temp--;
      }
      left = formula(temp + 1, caretPos - (temp + 1));

      temp = caretPos;
      temp++;
      if(formula[temp] == '(')
      {
         Int_t depth = 1;
         temp++;
         while(depth != 0 && temp < formula.Length())
         {
            if(formula[temp] == ')')
               depth--;
            if(formula[temp] == '(')
               depth++;
            temp++;
         }
         temp--;
      }
      while(temp < formula.Length() && !IsOperator(formula[temp]))
      {
         temp++;
      }
      right = formula(caretPos + 1, (temp - 1) - caretPos );

      TString pattern = TString::Format("%s^%s",left.Data(),right.Data());
      TString replacement = TString::Format("pow(%s,%s)",left.Data(),right.Data());
      formula.ReplaceAll(pattern,replacement);

      caretPos = formula.Last('^');
   }
}

void TFormula::HandleLinear(TString &formula)
{
   formula.ReplaceAll("++","@");
   Int_t linPos = formula.Index("@");
   Int_t NofLinParts = formula.CountChar((int)'@');
   fLinearParts.Expand(NofLinParts * 2);
   Int_t Nlinear = 0;
   while(linPos != kNPOS)
   {
      SetBit(kLinear,1);
      Int_t temp = linPos - 1;
      while(temp >= 0 && formula[temp] != '@')
      {
         temp--;
      }
      TString left = formula(temp+1,linPos - (temp +1));
      temp = linPos + 1;
      while(temp < formula.Length() && formula[temp] != '@')
      {
         temp++;
      }
      TString right = formula(linPos+1,temp - (linPos+1));
      TString pattern = TString::Format("%s@%s",left.Data(),right.Data());
      TString replacement = TString::Format("([%d]*(%s))+([%d]*(%s))",Nlinear,left.Data(),Nlinear+1,right.Data());
      formula.ReplaceAll(pattern,replacement);
      Nlinear += 2;
      TFormula *lin1 = new TFormula("__linear1",left);
      TFormula *lin2 = new TFormula("__linear2",right);
      lin1->SetBit(kNotGlobal,1);
      lin2->SetBit(kNotGlobal,1);
      gROOT->GetListOfFunctions()->Remove(lin1);
      gROOT->GetListOfFunctions()->Remove(lin2);
      fLinearParts.Add(lin1);
      fLinearParts.Add(lin2);
      linPos = formula.Index("@");
      delete lin1;
      delete lin2;
   }
}

void TFormula::PreProcessFormula(TString &formula)
{
   //*-*    
   //*-*    Preprocessing of formula
   //*-*    Replace all ** by ^, and removes spaces.
   //*-*    Handle also parametrized functions like polN,gaus,expo,landau
   //*-*    and exponentiation.
   //*-*    Similar functionality should be added here.
   //*-*    
   formula.ReplaceAll("**","^");
   formula.ReplaceAll(" ","");
   HandlePolN(formula);
   HandleParametrizedFunctions(formula);
   HandleExponentiation(formula);
   HandleLinear(formula);
}
Bool_t TFormula::PrepareFormula(TString &formula)
{
   fFuncs.clear();
   fReadyToExecute = false;
   ExtractFunctors(formula);
   fFuncs.sort();
   fFuncs.unique();

   ProcessFormula(formula);
   return fReadyToExecute;
}
void TFormula::ExtractFunctors(TString &formula)
{
   //*-*    
   //*-*    Extracts functors from formula, and put them in fFuncs.
   //*-*    Simple grammar:
   //*-*    <function>  := name(arg1,arg2...)
   //*-*    <variable>  := name
   //*-*    <parameter> := [number]
   //*-*    <name>      := String containing lower and upper letters, numbers, underscores
   //*-*    <number>    := Integer number
   //*-*    Operators are omitted.
   //*-*    
   TString name = "";
   TString body = "";
   for(Int_t i = 0 ; i < formula.Length(); ++i )
   {
      if(formula[i] == '[')
      {
         Int_t tmp = i;
         i++;
         TString param = "";
         while(formula[i] != ']' && i < formula.Length())
         {
            param.Append(formula[i++]);
         }
         i++;

         AddParameter(param,0);
         TString replacement = TString::Format("{[%s]}",param.Data());
         formula.Replace(tmp,i - tmp, replacement,replacement.Length());
         fFuncs.push_back(TFormulaFunction(param));
         continue;
      }
      if(isalpha(formula[i]) && !IsOperator(formula[i])) 
      {
         while( IsFunctionNameChar(formula[i]) && i < formula.Length())
         {
            name.Append(formula[i++]);
         }
         if(formula[i] == '(')
         {
            i++;
            if(formula[i] == ')')
            {
               fFuncs.push_back(TFormulaFunction(name,body,0));
               name = body = "";
               continue;  
            }
            Int_t depth = 1;
            Int_t args  = 1; // we will miss first argument
            while(depth != 0 && i < formula.Length())
            {
               switch(formula[i])
               {
                  case '(': depth++;   break;
                  case ')': depth--;   break;
                  case ',': if(depth == 1) args++; break;
               }
               if(depth != 0) // we don't want last ')' inside body
               {
                  body.Append(formula[i++]);
               }
            }
            Int_t originalBodyLen = body.Length();
            ExtractFunctors(body);
            formula.Replace(i-originalBodyLen,originalBodyLen,body,body.Length());
            i += body.Length() - originalBodyLen;
            fFuncs.push_back(TFormulaFunction(name,body,args));         
         }
         else
         {
            TString replacement = TString::Format("{%s}",name.Data());
            formula.Replace(i-name.Length(),name.Length(),replacement,replacement.Length());
            i += 2;
            fFuncs.push_back(TFormulaFunction(name));
         }
      }
      name = body = "";
      
   }
}
void TFormula::ProcessFormula(TString &formula)
{
   //*-*    
   //*-*    Iterates through funtors in fFuncs and performs the appropriate action.
   //*-*    If functor has 0 arguments (has only name) can be:
   //*-*     - variable
   //*-*       * will be replaced with x[num], where x is an array containing value of this variable under num.
   //*-*     - pre-defined formula
   //*-*       * will be replaced with formulas body
   //*-*     - constant
   //*-*       * will be replaced with constant value
   //*-*     - parameter
   //*-*       * will be replaced with p[num], where p is an array containing value of this parameter under num.
   //*-*    If has arguments it can be :
   //*-*     - function shortcut, eg. sin
   //*-*       * will be replaced with fullname of function, eg. sin -> TMath::Sin
   //*-*     - function from cling environment, eg. TMath::BreitWigner(x,y,z)
   //*-*       * first check if function exists, and has same number of arguments, then accept it and set as found.
   //*-*    If all functors after iteration are matched with corresponding action,
   //*-*    it inputs C++ code of formula into cling, and sets flag that formula is ready to evaluate.
   //*-*    

   for(list<TFormulaFunction>::iterator funcsIt = fFuncs.begin(); funcsIt != fFuncs.end(); ++funcsIt)
   {
      TFormulaFunction & fun = *funcsIt;
      if(fun.fFound)
         continue;
      if(fun.IsFuncCall())
      {
         map<TString,TString>::iterator it = fFunctionsShortcuts.find(fun.GetName());
         if(it != fFunctionsShortcuts.end())
         {
            TString shortcut = it->first;
            TString full = it->second;
            formula.ReplaceAll(shortcut,full);
            fun.fFound = true;
         }
         if(fun.GetName().Contains("::")) // restrict to only TMath?
         {
            TString className = fun.GetName()(0,fun.GetName()(0,fun.GetName().Index("::")).Length());
            TString functionName = fun.GetName()(fun.GetName().Index("::") + 2, fun.GetName().Length());
            Bool_t silent = true;
            TClass *tclass = new TClass(className,silent);
            TList *methodList = tclass->GetListOfAllPublicMethods();
            TIter next(methodList);
            TMethod *p;
            while ((p = (TMethod*) next()))
            {
               if (strcmp(p->GetName(),functionName.Data()) == 0 &&
                   p->GetNargs() == fun.GetNargs())
               { 
                  fun.fFound = true;
                  break;
               }
            }
         }
         if(!fun.fFound)
         {
            Error("TFormula","Could not find %s function with %d argument(s)",fun.GetName().Data(),fun.GetNargs());
         }
      }
      else
      {
         TFormula *old = (TFormula*)gROOT->GetListOfFunctions()->FindObject(fNamePrefix + fun.GetName());
         if(old)
         {
            fun.fFound = true;
            TString pattern = TString::Format("{%s}",fun.GetName().Data());
            TString replacement = old->GetExpFormula();
            PreProcessFormula(replacement);
            ExtractFunctors(replacement);
            formula.ReplaceAll(pattern,replacement);
            continue;
         }  
         map<TString,TFormulaVariable>::iterator varsIt = fVars.find(fun.GetName());
         if(varsIt!= fVars.end()) 
         {
            TString name = (*varsIt).second.GetName();
            Double_t value = (*varsIt).second.fValue;
            AddVariable(name,value);
            if(!fVars[name].fFound)
            {
               fVars[name].fFound = true;
               fNdim++;
            }
            TString pattern = TString::Format("{%s}",name.Data());   
            TString replacement = TString::Format("x[%d]",(*varsIt).second.fArrayPos);
            formula.ReplaceAll(pattern,replacement);
              
            fun.fFound = true; 
            continue;          
         }
         map<TString,Double_t>::iterator constIt = fConsts.find(fun.GetName());
         if(constIt != fConsts.end())
         {
            TString pattern = TString::Format("{%s}",fun.GetName().Data());
            TString value = TString::Format("%lf",(*constIt).second);
            formula.ReplaceAll(pattern,value);
            fun.fFound = true;
            continue;
         }
         
         map<TString,TFormulaVariable>::iterator paramsIt = fParams.find(fun.GetName());
         if(paramsIt != fParams.end())
         {
            TString name = (*paramsIt).second.GetName();
            TString pattern = TString::Format("{[%s]}",fun.GetName().Data());
            if(formula.Index(pattern) != kNPOS)
            {
               TString replacement = TString::Format("p[%d]",(*paramsIt).second.fArrayPos);
               formula.ReplaceAll(pattern,replacement);
               
            }
            fun.fFound = true;
            continue;
         }
         fun.fFound = false;
      }
   }
   Bool_t allFunctorsMatched = true;
   for(list<TFormulaFunction>::iterator it = fFuncs.begin(); it != fFuncs.end(); it++)
   {
      if(!it->fFound)
      {
         allFunctorsMatched = false;
         break;
      }
   }
   
   if(!fReadyToExecute && allFunctorsMatched)
   {
      fReadyToExecute = true;
      Bool_t hasVariables = (fNdim > 0);
      Bool_t hasParameters = (fNpar > 0);
      Bool_t hasBoth = hasVariables && hasParameters;
      Bool_t inputIntoCling = (formula.Length() > 0);
      TString argumentsPrototype = 
         TString::Format("%s%s%s",(hasVariables ? "Double_t *x" : ""), (hasBoth ? "," : ""),
                        (hasParameters  ? "Double_t *p" : ""));
      fClingInput = TString::Format("Double_t %s(%s){ return %s ; }", fName.Data(),argumentsPrototype.Data(),formula.Data());

      if(inputIntoCling)
      {
         InputFormulaIntoCling();
      }
      else
      {
         fReadyToExecute = true;
         fAllParametersSetted = true;
         fClingInitialized = true;
      }

   }
}
const TObject* TFormula::GetLinearPart(Int_t i)
{
   // Return linear part.

   if (!fLinearParts.IsEmpty())
      return fLinearParts.UncheckedAt(i);
   return 0;
}
void TFormula::AddVariable(const TString &name, Double_t value)
{
   //*-*    
   //*-*    Adds variable to known variables, and reprocess formula.
   //*-*    
   
   if(fVars.find(name) != fVars.end() )
   {
      TFormulaVariable & var = fVars[name];
      var.fValue = value;

      if(var.fArrayPos < 0)
      {
         var.fArrayPos = fVars.size();
      }
      if(var.fArrayPos >= (int)fClingVariables.capacity())
      {
         fClingVariables.reserve(2 * fClingVariables.capacity());
      }
      fClingVariables[var.fArrayPos] = value;
   }
   else
   {
      TFormulaVariable var(name,value,fVars.size());
      fVars[name] = var;
      fClingVariables.push_back(value);
      ProcessFormula(fClingInput);
   }

}
void TFormula::AddVariables(const pair<TString,Double_t> *vars, const Int_t size)
{
   //*-*    
   //*-*    Adds multiple variables.
   //*-*    First argument is an array of pairs<TString,Double>, where
   //*-*    first argument is name of variable,
   //*-*    second argument represents value.
   //*-*    size - number of variables passed in first argument
   //*-*    

   Bool_t anyNewVar = false;
   for(Int_t i = 0 ; i < size; ++i)
   {

      pair<TString,Double_t> v = vars[i];

      TFormulaVariable &var = fVars[v.first];
      if(var.fArrayPos < 0)
      {

         var.fName = v.first;
         var.fArrayPos = fVars.size();
         anyNewVar = true;
         var.fValue = v.second;
         if(var.fArrayPos >= (int)fClingVariables.capacity())
         {
            Int_t multiplier = 2;
            if(fFuncs.size() > 100)
            {
               multiplier = TMath::Floor(TMath::Log10(fFuncs.size()) * 10);
            }
            fClingVariables.reserve(multiplier * fClingVariables.capacity());
         }
         fClingVariables.push_back(v.second);
      }
      else
      {
         var.fValue = v.second;
         fClingVariables[var.fArrayPos] = v.second;
      }
   }
   if(anyNewVar)
   {
      ProcessFormula(fClingInput);
   }

}
void TFormula::SetVariables(const pair<TString,Double_t> *vars, const Int_t size)
{
   //*-*    
   //*-*    Sets multiple variables.
   //*-*    First argument is an array of pairs<TString,Double>, where
   //*-*    first argument is name of variable,
   //*-*    second argument represents value.
   //*-*    size - number of variables passed in first argument
   //*-*
   for(Int_t i = 0; i < size; ++i)
   {
      pair<TString,Double_t> v = vars[i];
      if(fVars.find(v.first) != fVars.end())
      {
         fVars[v.first].fValue = v.second;
         fClingVariables[fVars[v.first].fArrayPos] = v.second;
      }  
      else
      {
         Error("SetVariables","Variable %s is not defined.",v.first.Data());
      }
   }
}

Double_t TFormula::GetVariable(const TString &name)
{
   //*-*    
   //*-*    Returns variable value.
   //*-*    
   if(fVars.find(name) == fVars.end())
   {
      Error("GetVariable","Variable %s is not defined.",name.Data());
      return -1;
   }
   return fVars[name].fValue;
}
void TFormula::SetVariable(const TString &name, Double_t value)
{
   //*-*    
   //*-*    Sets variable value.
   //*-*    
   if(fVars.find(name) == fVars.end())
   {
      Error("SetVariable","Variable %s is not defined.",name.Data());
      return;
   }
   fVars[name].fValue = value;
   fClingVariables[fVars[name].fArrayPos] = value;
}

void TFormula::AddParameter(const TString &name, Double_t value)
{
   //*-*    
   //*-*    Adds parameter to known parameters.
   //*-*    User should use SetParameter, because parameters are added during initialization part,
   //*-*    and after that adding new will be pointless.
   //*-*    

   if(fParams.find(name) != fParams.end() )
   {
      TFormulaVariable & par = fParams[name];
      par.fValue = value;
      if (par.fArrayPos < 0)
      {
         par.fArrayPos = fParams.size();
      }
      if(par.fArrayPos >= (int)fClingParameters.capacity())
      {
         fClingParameters.reserve(2 * fClingParameters.capacity());
      }
      fClingParameters[par.fArrayPos] = value;
   }
   else
   {
      fNpar++;
      TFormulaVariable par(name,value,fParams.size());
      fParams[name] = par;
      fClingParameters.push_back(value);
   }

} 
Double_t TFormula::GetParameter(const TString &name)
{
   //*-*    
   //*-*    Returns parameter value.
   //*-*    
   if(fParams.find(name) == fParams.end())
   {
      Error("GetParameter","Parameter %s is not defined.",name.Data());
      return -1;
   }
   return fParams[name].fValue;
}
Double_t TFormula::GetParameter(Int_t param)
{
   TString name = TString::Format("%d",param);
   return GetParameter(name);
}
const char* TFormula::GetParName(Int_t ipar) const
{
   return TString::Format("p%d",ipar).Data();
}
Double_t* TFormula::GetParameters() const
{
   if(!fClingParameters.empty())
      return const_cast<Double_t*>(&fClingParameters[0]);
   return 0;
}

void TFormula::GetParameters(Double_t *params)
{
   for(Int_t i = 0; i < fNpar; ++i)
   {
      params[i] = fClingParameters[i];
   }
}
void TFormula::SetParameter(const TString &name, Double_t value)
{
   //*-*    
   //*-*    Sets parameter value.
   //*-*    
   if(fParams.find(name) == fParams.end())
   {
      Error("SetParameter","Parameter %s is not defined.",name.Data());
      return;
   }
   fParams[name].fValue = value;
   fParams[name].fFound = true;
   fClingParameters[fParams[name].fArrayPos] = value;
   fAllParametersSetted = true;
   for(map<TString,TFormulaVariable>::iterator it = fParams.begin(); it != fParams.end(); it++)
   {
      if(!it->second.fFound)
      {
         fAllParametersSetted = false;
         break;
      }
   }
}
void TFormula::SetParameters(const pair<TString,Double_t> *params,const Int_t size)
{
   //*-*    
   //*-*    Set multiple parameters.
   //*-*    First argument is an array of pairs<TString,Double>, where
   //*-*    first argument is name of parameter,
   //*-*    second argument represents value.
   //*-*    size - number of params passed in first argument
   //*-*    
   for(Int_t i = 0 ; i < size ; ++i)
   {
      pair<TString,Double_t> p = params[i];
      if(fParams.find(p.first) == fParams.end())
      {
         Error("SetParameters","Parameter %s is not defined",p.first.Data());
         continue;
      }
      fParams[p.first].fValue = p.second;
      fParams[p.first].fFound = true;
      fClingParameters[fParams[p.first].fArrayPos] = p.second;
   }
   fAllParametersSetted = true;
   for(map<TString,TFormulaVariable>::iterator it = fParams.begin(); it != fParams.end(); it++)
   {
      if(!it->second.fFound)
      {
         fAllParametersSetted = false;
         break;
      }
   }
}
void TFormula::SetParameters(const Double_t *params)
{
   SetParameters(params,fNpar);
}
void TFormula::SetParameters(Double_t p0,Double_t p1,Double_t p2,Double_t p3,Double_t p4,
                   Double_t p5,Double_t p6,Double_t p7,Double_t p8,
                   Double_t p9,Double_t p10)
{
   if(fNpar >= 1) SetParameter(0,p0);
   if(fNpar >= 2) SetParameter(1,p1);
   if(fNpar >= 3) SetParameter(2,p2);
   if(fNpar >= 4) SetParameter(3,p3);
   if(fNpar >= 5) SetParameter(4,p4);
   if(fNpar >= 6) SetParameter(5,p5);
   if(fNpar >= 7) SetParameter(6,p6);
   if(fNpar >= 8) SetParameter(7,p7);
   if(fNpar >= 9) SetParameter(8,p8);
   if(fNpar >= 10) SetParameter(9,p9);
   if(fNpar >= 11) SetParameter(10,p10);
}
void TFormula::SetParameter(Int_t param, Double_t value)
{
   TString name = TString::Format("%d",param);
   SetParameter(name,value);
}
void TFormula::SetParNames(const char *name0,const char *name1,const char *name2,const char *name3,
                 const char *name4, const char *name5,const char *name6,const char *name7,
                 const char *name8,const char *name9,const char *name10)
{
   SetParName(0,name0);
   SetParName(1,name1);
   SetParName(2,name2);
   SetParName(3,name3);
   SetParName(4,name4);
   SetParName(5,name5);
   SetParName(6,name6);
   SetParName(7,name7);
   SetParName(8,name8);
   SetParName(9,name9);
   SetParName(10,name10);
}
void TFormula::SetParName(Int_t ipar, const char * name)
{
   //TODO
}
void TFormula::SetParameters(const Double_t *params, Int_t size)
{
   if(!params || size < 0 || size > fNpar) return;
   for(Int_t i = 0; i < size; ++i)
   {
      TString name = TString::Format("%d",i);
      SetParameter(name,params[i]);
   }
}
Double_t TFormula::EvalPar(const Double_t *x,const Double_t *params)
{
   SetParameters(params);
   if(fNdim >= 1) SetVariable("x",x[0]);
   if(fNdim >= 2) SetVariable("y",x[1]);
   if(fNdim >= 3) SetVariable("z",x[2]);
   if(fNdim >= 4) SetVariable("t",x[3]);

   return Eval();
}
Double_t TFormula::Eval(Double_t x, Double_t y, Double_t z, Double_t t)
{
   //*-*    
   //*-*    Sets 4 default variables x, y, z, t and evaluate formula.
   //*-*    
   if(fNdim >= 1) SetVariable("x",x);
   if(fNdim >= 2) SetVariable("y",y);
   if(fNdim >= 3) SetVariable("z",z);
   if(fNdim >= 4) SetVariable("t",t);
   return Eval();
}
Double_t TFormula::Eval(Double_t x, Double_t y , Double_t z)
{
   //*-*    
   //*-*    Sets 3 default variables x, y, z and evaluate formula.
   //*-*    

   if(fNdim >= 1) SetVariable("x",x);
   if(fNdim >= 2) SetVariable("y",y);
   if(fNdim >= 3) SetVariable("z",z);

   return Eval();
}
Double_t TFormula::Eval(Double_t x, Double_t y)
{
   //*-*    
   //*-*    Sets 2 default variables x, y and evaluate formula.
   //*-*    

   if(fNdim >= 1) SetVariable("x",x);
   if(fNdim >= 2) SetVariable("y",y);
   return Eval();
}
Double_t TFormula::Eval(Double_t x)
{
   //*-*    
   //*-*    Sets 1 default variable x and evaluate formula.
   //*-*    

   SetVariable("x",x);
   return Eval();
}
Double_t TFormula::Eval()
{
   //*-*    
   //*-*    Evaluate formula.
   //*-*    If formula is not ready to execute(missing parameters/variables), 
   //*-*    print these which are not known.
   //*-*    If parameter has default value, and has not been setted, appropriate warning is shown.
   //*-*    

   if(!fReadyToExecute)
   {
      Error("Eval","Formula not ready to execute. Missing parameters/variables");
      for(list<TFormulaFunction>::iterator it = fFuncs.begin(); it != fFuncs.end(); ++it)
      {
         TFormulaFunction fun = *it;
         if(!fun.fFound)
         {
            printf("%s is uknown.\n",fun.GetName().Data());
         }
      }
      return -1;
   }
   if(!fAllParametersSetted)
   {
      Warning("Eval","Not all parameters are setted.");
      for(map<TString,TFormulaVariable>::iterator it = fParams.begin(); it != fParams.end(); ++it)
      {
         pair<TString,TFormulaVariable> param = *it;
         if(!param.second.fFound)
         {
            printf("%s has default value %lf\n",param.first.Data(),param.second.GetValue());
         }
      }  

   }
   Double_t result = 0;
   fMethod->Execute(result);
   return result;
}