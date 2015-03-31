// @(#)root/hist:$Id$
// Author: Nicolas Brun   19/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- Formula.h

#ifndef ROOT_v5_TFormula
#define ROOT_v5_TFormula



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFormula                                                             //
//                                                                      //
// The old formula base class  f(x,y,z,par)                                 //
// mantained for backward compatibility and TTree usage                  //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TBits
#include "TBits.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif


const Int_t kMAXFOUND = 500;
const Int_t kTFOperMask = 0x7fffff;
const UChar_t kTFOperShift = 23;



namespace ROOT {
   namespace v5 {

     class TFormulaPrimitive;

class  TOperOffset {
   friend class TFormula;
public:
   enum {
      kVariable  = 0,
      kParameter = 1,
      kConstant  = 2
   };
   TOperOffset();
protected:
   Short_t fType0;            // type     of operand  0
   Short_t fOffset0;          // offset   of operand  0
   Short_t fType1;            // type     of operand  1
   Short_t fOffset1;          // offset   of operand  1
   Short_t fType2;            // type     of operand  2
   Short_t fOffset2;          // offset   of operand  2
   Short_t fType3;            // type     of operand  3
   Short_t fOffset3;          // offset   of operand  3
   Short_t fToJump;           // where to jump in case of optimized boolen
   Short_t fOldAction;        // temporary variable used during optimization
};

      
class TFormula : public TNamed {

protected:

   typedef Double_t (TObject::*TFuncG)(const Double_t*,const Double_t*) const;

   Int_t      fNdim;            //Dimension of function (1=1-Dim, 2=2-Dim,etc)
   Int_t      fNpar;            //Number of parameters
   Int_t      fNoper;           //Number of operators
   Int_t      fNconst;          //Number of constants
   Int_t      fNumber;          //formula number identifier
   Int_t      fNval;            //Number of different variables in expression
   Int_t      fNstring;         //Number of different constants character strings
   TString   *fExpr;            //[fNoper] List of expressions
private:
   Int_t     *fOper;            //[fNoper] List of operators. (See documentation for changes made at version 7)
protected:
   Double_t  *fConst;           //[fNconst] Array of fNconst formula constants
   Double_t  *fParams;          //[fNpar] Array of fNpar parameters
   TString   *fNames;           //[fNpar] Array of parameter names
   TObjArray  fFunctions;       //Array of function calls to make
   TObjArray  fLinearParts;     //Linear parts if the formula is linear (contains '|' or "++")

   TBits      fAlreadyFound;    //! cache for information

   // Optimized expression
   Int_t                fNOperOptimized; //!Number of operators after optimization
   TString             *fExprOptimized;  //![fNOperOptimized] List of expressions
   Int_t               *fOperOptimized;  //![fNOperOptimized] List of operators. (See documentation for changes made at version 7)
   TOperOffset         *fOperOffset;     //![fNOperOptimized]         Offsets of operrands
   TFormulaPrimitive  **fPredefined;      //![fNPar] predefined function
   TFuncG               fOptimal; //!pointer to optimal function

   Int_t             PreCompile();
   virtual Bool_t    CheckOperands(Int_t operation, Int_t &err);
   virtual Bool_t    CheckOperands(Int_t leftoperand, Int_t rightoperartion, Int_t &err);
   virtual Bool_t    StringToNumber(Int_t code);
   void              MakePrimitive(const char *expr, Int_t pos);
   inline Int_t     *GetOper() const { return fOper; }
   inline Short_t    GetAction(Int_t code) const { return fOper[code] >> kTFOperShift; }
   inline Int_t      GetActionParam(Int_t code) const { return fOper[code] & kTFOperMask; }

   inline void       SetAction(Int_t code, Int_t value, Int_t param = 0) {
      fOper[code]  = (value) << kTFOperShift;
      fOper[code] += param;
   }
   inline Int_t     *GetOperOptimized() const { return fOperOptimized; }
   inline Short_t    GetActionOptimized(Int_t code) const { return fOperOptimized[code] >> kTFOperShift; }
   inline Int_t      GetActionParamOptimized(Int_t code) const { return fOperOptimized[code] & kTFOperMask; }

   inline void       SetActionOptimized(Int_t code, Int_t value, Int_t param = 0) {
      fOperOptimized[code]  = (value) << kTFOperShift;
      fOperOptimized[code] += param;
   }

   void            ClearFormula(Option_t *option="");
   virtual Bool_t  IsString(Int_t oper) const;

   virtual void    Convert(UInt_t fromVersion);
   //
   // Functions  - used for formula evaluation
   Double_t        EvalParFast(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive0(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive1(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive2(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive3(const Double_t *x, const Double_t *params);
   Double_t        EvalPrimitive4(const Double_t *x, const Double_t *params);

   // Action code for Version 6 and above.
   enum {
      kEnd      = 0,
      kAdd      = 1, kSubstract = 2,
      kMultiply = 3, kDivide    = 4,
      kModulo   = 5,

      kcos      = 10, ksin  = 11 , ktan  = 12,
      kacos     = 13, kasin = 14 , katan = 15,
      katan2    = 16,
      kfmod     = 17,

      kpow      = 20, ksq = 21, ksqrt     = 22,

      kstrstr   = 23,

      kmin      = 24, kmax = 25,

      klog      = 30, kexp = 31, klog10 = 32,

      kpi     = 40,

      kabs    = 41 , ksign= 42,
      kint    = 43 ,
      kSignInv= 44 ,
      krndm   = 50 ,

      kAnd      = 60, kOr          = 61,
      kEqual    = 62, kNotEqual    = 63,
      kLess     = 64, kGreater     = 65,
      kLessThan = 66, kGreaterThan = 67,
      kNot      = 68,

      kcosh   = 70 , ksinh  = 71, ktanh  = 72,
      kacosh  = 73 , kasinh = 74, katanh = 75,

      kStringEqual = 76, kStringNotEqual = 77,

      kBitAnd    = 78, kBitOr     = 79,
      kLeftShift = 80, kRightShift = 81,

      kJumpIf = 82, kJump = 83,

      kexpo   = 100 , kxexpo   = 100, kyexpo   = 101, kzexpo   = 102, kxyexpo   = 105,
      kgaus   = 110 , kxgaus   = 110, kygaus   = 111, kzgaus   = 112, kxygaus   = 115,
      klandau = 120 , kxlandau = 120, kylandau = 121, kzlandau = 122, kxylandau = 125,
      kpol    = 130 , kxpol    = 130, kypol    = 131, kzpol    = 132,

      kParameter       = 140,
      kConstant        = 141,
      kBoolOptimize    = 142,
      kStringConst     = 143,
      kVariable        = 144,
      kFunctionCall    = 145,
      kData            = 146,
      kUnary           = 147,
      kBinary          = 148,
      kThree           = 149,
      kDefinedVariable = 150,
      kDefinedString   = 151,
      //
      kPlusD           = 152,
      kPlusDD          = 153,
      kMultD           = 154,
      kMultDD          = 155,
      kBoolOptimizeOr  = 156,
      kBoolOptimizeAnd = 157,
      kBoolSet         = 158,
      kFDM             = 159,
      kFD0             = 160,
      kFD1             = 161,
      kFD2             = 162,
      kFD3             = 163
   };

public:
   // TFormula status bits
   enum {
      kNotGlobal     = BIT(10),  // don't store in gROOT->GetListOfFunction
      kNormalized    = BIT(14),   // set to true if the function (ex gausn) is normalized
      kLinear        = BIT(16)    //set to true if the function is for linear fitting
   };

               TFormula();
               TFormula(const char *name,const char *formula);
               TFormula(const TFormula &formula);
   TFormula&   operator=(const TFormula &rhs);
   virtual    ~TFormula();

 public:
   void                Optimize();
   virtual void        Analyze(const char *schain, Int_t &err, Int_t offset=0);
   virtual Bool_t      AnalyzeFunction(TString &chaine, Int_t &err, Int_t offset=0);
   virtual Int_t       Compile(const char *expression="");
   virtual void        Copy(TObject &formula) const;
   virtual void        Clear(Option_t *option="");
   virtual char       *DefinedString(Int_t code);
   virtual Double_t    DefinedValue(Int_t code);
   virtual Int_t       DefinedVariable(TString &variable,Int_t &action);
   virtual Double_t    Eval(Double_t x, Double_t y=0, Double_t z=0, Double_t t=0) const;
   virtual Double_t    EvalParOld(const Double_t *x, const Double_t *params=0);
   virtual Double_t    EvalPar(const Double_t *x, const Double_t *params=0){return ((*this).*fOptimal)(x,params);};
   virtual const TObject *GetLinearPart(Int_t i);
   virtual Int_t       GetNdim() const {return fNdim;}
   virtual Int_t       GetNpar() const {return fNpar;}
   virtual Int_t       GetNumber() const {return fNumber;}
   virtual TString     GetExpFormula(Option_t *option="") const;
   Double_t            GetParameter(Int_t ipar) const;
   Double_t            GetParameter(const char *name) const;
   virtual Double_t   *GetParameters() const {return fParams;}
   virtual void        GetParameters(Double_t *params){for(Int_t i=0;i<fNpar;i++) params[i] = fParams[i];}
   virtual const char *GetParName(Int_t ipar) const;
   virtual Int_t       GetParNumber(const char *name) const;
   virtual Bool_t      IsLinear() const {return TestBit(kLinear);}
   virtual Bool_t      IsNormalized() const {return TestBit(kNormalized);}
   virtual void        Print(Option_t *option="") const; // *MENU*
   virtual void        ProcessLinear(TString &replaceformula);
   virtual void        SetNumber(Int_t number) {fNumber = number;}
   virtual void        SetParameter(const char *name, Double_t parvalue);
   virtual void        SetParameter(Int_t ipar, Double_t parvalue);
   virtual void        SetParameters(const Double_t *params);
   virtual void        SetParameters(Double_t p0,Double_t p1,Double_t p2=0,Double_t p3=0,Double_t p4=0,
                                     Double_t p5=0,Double_t p6=0,Double_t p7=0,Double_t p8=0,
                                     Double_t p9=0,Double_t p10=0); // *MENU*
   virtual void        SetParName(Int_t ipar, const char *name);
   virtual void        SetParNames(const char *name0="p0",const char *name1="p1",const char
                                   *name2="p2",const char *name3="p3",const char
                                   *name4="p4", const char *name5="p5",const char *name6="p6",const char *name7="p7",const char
                                   *name8="p8",const char *name9="p9",const char *name10="p10"); // *MENU*
   virtual void        Update() {;}

   static  void        SetMaxima(Int_t maxop=1000, Int_t maxpar=1000, Int_t maxconst=1000);
   static  void        GetMaxima(Int_t& maxop, Int_t& maxpar, Int_t& maxconst);

   void Streamer(TBuffer &b, const TClass *onfile_class);
   void Streamer(TBuffer &b, Int_t version, UInt_t start, UInt_t count, const TClass *onfile_class = 0);

   ClassDef(ROOT::v5::TFormula,8)  //The formula base class  f(x,y,z,par)
};

   } // end namespace v5

} // end namespace ROOT
      
#endif
