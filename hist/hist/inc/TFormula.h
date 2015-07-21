
// @(#)root/hist:$Id$
// Author: Maciej Zimnoch   30/09/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- TFormula.h
#ifndef ROOT_TFormula
#define ROOT_TFormula


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
 #ifndef ROOT_TBits
#include "TBits.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif
#include "TMethodCall.h"
#include "TInterpreter.h"
#include <vector>
#include <list>
#include <map>

class TFormulaFunction
{
public:
   TString  fName;
   TString  fBody;
   Int_t    fNargs;
   Bool_t   fFound;
   Bool_t   fFuncCall;
   const char *  GetName() const    { return fName.Data(); }
   const char *  GetBody() const    { return fBody.Data(); }
   Int_t    GetNargs() const   { return fNargs;}
   Bool_t   IsFuncCall() const { return fFuncCall;}
   TFormulaFunction(){}
   TFormulaFunction(const TString &name, const TString &body, int numArgs)
      : fName(name),fBody(body),fNargs(numArgs),fFound(false),fFuncCall(true) {}
   TFormulaFunction(const TString& name)
   : fName(name),fBody(""),fNargs(0),fFound(false),fFuncCall(false){}
   Bool_t operator<(const TFormulaFunction &rhv) const
   {
      // order by length - first the longer ones to avoid replacing wrong functions 
      if ( fName.Length() > rhv.fName.Length() )
         return true;
      else if ( fName.Length() > rhv.fName.Length() )
         return false;
      // case of equal length
      return fName < rhv.fName && fBody < rhv.fBody;
   }
   Bool_t operator==(const TFormulaFunction &rhv) const
   {
      return fName == rhv.fName && fBody == rhv.fBody && fNargs == rhv.fNargs;
   }
};
class TFormulaVariable
{
public:
   TString fName;
   Double_t fValue;
   Int_t fArrayPos;
   Bool_t fFound;
   const char * GetName() const     { return fName.Data(); }
   Double_t GetInitialValue() const    { return fValue; }
   Int_t    GetArrayPos() const { return fArrayPos; }
   TFormulaVariable():fName(""),fValue(-1),fArrayPos(-1),fFound(false){}
   TFormulaVariable(const TString &name, Double_t value, Int_t pos)
   : fName(name), fValue(value), fArrayPos(pos),fFound(false) {}
   Bool_t operator<(const TFormulaVariable &rhv) const
   {
      return fName < rhv.fName;
   }
};

// Functor defining the parameter order
struct TFormulaParamOrder {
   // comparison function
   bool operator() (const TString& a, const TString& b) const;
};


class TFormula : public TNamed
{
private:

   // All data members are transient apart from the string defining the formula and the parameter values

   TString           fClingInput;           //! input function passed to Cling
   std::vector<Double_t>  fClingVariables;       //!  cached variables
   std::vector<Double_t>  fClingParameters;      //  parameter values
   Bool_t            fReadyToExecute;       //!
   Bool_t            fClingInitialized;  //!  transient to force re-initialization
   Bool_t            fAllParametersSetted;    // flag to control if all parameters are setted
   TMethodCall*      fMethod;        //! pointer to methocall
   TString           fClingName;     //! unique name passed to Cling to define the function ( double clingName(double*x, double*p) )

   TInterpreter::CallFuncIFacePtr_t::Generic_t fFuncPtr;   //!  function pointer

   void     InputFormulaIntoCling();
   Bool_t   PrepareEvalMethod();
   void     FillDefaults();
   void     HandlePolN(TString &formula);
   void     HandleParametrizedFunctions(TString &formula);
   void     HandleExponentiation(TString &formula);
   void     HandleLinear(TString &formula);
   static Bool_t   IsDefaultVariableName(const TString &name);
protected:

   std::list<TFormulaFunction>         fFuncs;    //!
   std::map<TString,TFormulaVariable>  fVars;     //!  list of  variable names
   std::map<TString,Int_t,TFormulaParamOrder>   fParams;   //  list of  parameter names
   std::map<TString,Double_t>          fConsts;   //!
   std::map<TString,TString>           fFunctionsShortcuts;  //!
   TString                        fFormula;
   Int_t                          fNdim;  //!
   Int_t                          fNpar;  //!
   Int_t                          fNumber;  //!
   std::vector<TObject*>          fLinearParts;  // vector of linear functions

   static Bool_t IsOperator(const char c);
   static Bool_t IsBracket(const char c);
   static Bool_t IsFunctionNameChar(const char c);
   static Bool_t IsScientificNotation(const TString & formula, int ipos);
   static Bool_t IsHexadecimal(const TString & formula, int ipos);
   void   ExtractFunctors(TString &formula);
   void   PreProcessFormula(TString &formula);
   void   ProcessFormula(TString &formula);
   Bool_t PrepareFormula(TString &formula);
   void   ReplaceParamName(TString &formula, const TString & oldname, const TString & name);
   void   DoAddParameter(const TString &name, Double_t value, bool processFormula);
   void   DoSetParameters(const Double_t * p, Int_t size);
   void   SetPredefinedParamNames(); 

   Double_t       DoEval(const Double_t * x, const Double_t * p = nullptr) const;

   enum {
      kNotGlobal     = BIT(10),  // don't store in gROOT->GetListOfFunction (it should be protected)
   };

public:

   enum {
      kNormalized    = BIT(14),   // set to true if the TFormula (ex gausn) is normalized
      kLinear        = BIT(16)    //set to true if the TFormula is for linear fitting
   };
                  TFormula();
   virtual        ~TFormula();
   TFormula&      operator=(const TFormula &rhs);
   TFormula(const char *name, const char * formula = "", bool addToGlobList = true);
                  TFormula(const TFormula &formula);
   //               TFormula(const char *name, Int_t nparams, Int_t ndims);

   void           AddParameter(const TString &name, Double_t value = 0) { DoAddParameter(name,value,true); }
   void           AddVariable(const TString &name, Double_t value = 0);
   void           AddVariables(const TString *vars, const Int_t size);
   Int_t          Compile(const char *expression="");
   virtual void   Copy(TObject &f1) const;
   virtual void   Clear(Option_t * option="");
   Double_t       Eval(Double_t x) const;
   Double_t       Eval(Double_t x, Double_t y) const;
   Double_t       Eval(Double_t x, Double_t y , Double_t z) const;
   Double_t       Eval(Double_t x, Double_t y , Double_t z , Double_t t ) const;
   Double_t       EvalPar(const Double_t *x, const Double_t *params=0) const;
   TString        GetExpFormula(Option_t *option="") const;
   const TObject *GetLinearPart(Int_t i) const;
   Int_t          GetNdim() const {return fNdim;}
   Int_t          GetNpar() const {return fNpar;}
   Int_t          GetNumber() const { return fNumber; }
   const char *   GetParName(Int_t ipar) const;
   Int_t          GetParNumber(const char * name) const;
   Double_t       GetParameter(const char * name) const;
   Double_t       GetParameter(Int_t param) const;
   Double_t*      GetParameters() const;
   void           GetParameters(Double_t *params) const;
   Double_t       GetVariable(const char *name) const;
   Int_t          GetVarNumber(const char *name) const;
   TString        GetVarName(Int_t ivar) const;
   Bool_t         IsValid() const { return fReadyToExecute && fClingInitialized; }
   Bool_t         IsLinear() const { return TestBit(kLinear); }
   void           Print(Option_t *option = "") const;
   void           SetName(const char* name);
   void           SetParameter(const char* name, Double_t value);
   void           SetParameter(Int_t param, Double_t value);
   void           SetParameters(const Double_t *params);
   //void           SetParameters(const pair<TString,Double_t> *params, const Int_t size);
   void           SetParameters(Double_t p0,Double_t p1,Double_t p2=0,Double_t p3=0,Double_t p4=0,
                                     Double_t p5=0,Double_t p6=0,Double_t p7=0,Double_t p8=0,
                                     Double_t p9=0,Double_t p10=0); // *MENU*
   void           SetParName(Int_t ipar, const char *name);
   void           SetParNames(const char *name0="p0",const char *name1="p1",const char
                             *name2="p2",const char *name3="p3",const char
                             *name4="p4", const char *name5="p5",const char *name6="p6",const char *name7="p7",const char
                             *name8="p8",const char *name9="p9",const char *name10="p10"); // *MENU*
   void           SetVariable(const TString &name, Double_t value);
   void           SetVariables(const std::pair<TString,Double_t> *vars, const Int_t size);

   ClassDef(TFormula,10)
};
#endif
