// @(#)root/treeplayer:$Id$
// Author: Rene Brun   19/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- TreeFormula.h

#ifndef ROOT_TTreeFormula
#define ROOT_TTreeFormula



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFormula                                                         //
//                                                                      //
// The Tree formula class                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "v5/TFormula.h"

#include "TLeaf.h"

#include "TObjArray.h"

#include <string>
#include <vector>

#ifdef R__OLDHPACC
namespace std {
   using ::string;
   using ::vector;
}
#endif

const Int_t kMAXCODES = kMAXFOUND; // must be the same as kMAXFOUND in TFormulaOld
const Int_t kMAXFORMDIM = 5; // Maximum number of array dimensions support in TTreeFormula

class TTree;
class TArrayI;
class TMethodCall;
class TLeafObject;
class TDataMember;
class TStreamerElement;
class TFormLeafInfoMultiVarDim;
class TFormLeafInfo;
class TBranchElement;
class TAxis;
class TTreeFormulaManager;


class TTreeFormula : public ROOT::v5::TFormula {

friend class TTreeFormulaManager;

protected:
   enum EStatusBits {
      kIsCharacter = BIT(12),
      kMissingLeaf = BIT(15), // true if some of the needed leaves are missing in the current TTree
      kIsInteger   = BIT(17), // true if the branch contains an integer variable
      kNeedEntries = BIT(18)  // true if the formula uses Entries$
   };
   enum {
      kDirect, kDataMember, kMethod,
      kIndexOfEntry, kEntries, kLength, kIteration, kLengthFunc, kSum, kEntryList,
      kTreeMember,
      kIndexOfLocalEntry,
      kMin, kMax,
      kLocalEntries

   };
   enum {
      kAlias           = 200,
      kAliasString     = 201,
      kAlternate       = 202,
      kAlternateString = 203,
      kMinIf           = 204,
      kMaxIf           = 205
   };

   // Helper struct to hold a cache
   // that can accelerate calculation of the RealIndex.
   struct RealInstanceCache {
      Int_t fInstanceCache = 0;
      Int_t fLocalIndexCache = 0;
      Int_t fVirtAccumCache = 0;
   };

   TTree      *fTree;             //! pointer to Tree
   Int_t       fCodes[kMAXCODES]; //  List of leaf numbers referenced in formula
   Int_t       fNdata[kMAXCODES]; //! This caches the physical number of element in the leaf or data member.
   Int_t       fNcodes;           //  Number of leaves referenced in formula
   Bool_t      fHasCast;          //  Record whether the formula contain a cast operation or not
   Int_t       fMultiplicity;     //  Indicator of the variability of the formula
   Int_t       fNindex;           //  Size of fIndex
   Int_t      *fLookupType;       //[fNindex] array indicating how each leaf should be looked-up
   TObjArray   fLeaves;           //!  List of leaf used in this formula.
   TObjArray   fDataMembers;      //!  List of leaf data members
   TObjArray   fMethods;          //!  List of leaf method calls
   TObjArray   fExternalCuts;     //!  List of TCutG and TEntryList used in the formula
   TObjArray   fAliases;          //!  List of TTreeFormula for each alias used.
   TObjArray   fLeafNames;        //   List of TNamed describing leaves
   TObjArray   fBranches;         //!  List of branches to read.  Similar to fLeaves but duplicates are zeroed out.
   Bool_t      fQuickLoad;        //!  If true, branch GetEntry is only called when the entry number changes.
   Bool_t      fNeedLoading;      //!  If true, the current entry has not been loaded yet.

   Int_t       fNdimensions[kMAXCODES];              //Number of array dimensions in each leaf
   Int_t       fFixedSizes[kMAXCODES][kMAXFORMDIM];  //Physical sizes of lower dimensions for each leaf
   UChar_t     fHasMultipleVarDim[kMAXCODES];        //True if the corresponding variable is an array with more than one variable dimension.

   //the next line should have a mutable in front. See GetNdata()
   Int_t       fCumulSizes[kMAXCODES][kMAXFORMDIM];  //Accumulated sizes of lower dimensions for each leaf after variable dimensions has been calculated
   Int_t       fIndexes[kMAXCODES][kMAXFORMDIM];     //Index of array selected by user for each leaf
   TTreeFormula *fVarIndexes[kMAXCODES][kMAXFORMDIM];  //Pointer to a variable index.

   TAxis                    *fAxis;           //! pointer to histogram axis if this is a string
   Bool_t                    fDidBooleanOptimization;  //! True if we executed one boolean optimization since the last time instance number 0 was evaluated
   TTreeFormulaManager      *fManager;        //! The dimension coordinator.

   // Helper members and function used during the construction and parsing
   TList                    *fDimensionSetup; //! list of dimension setups, for delayed creation of the dimension information.
   std::vector<std::string>  fAliasesUsed;    //! List of aliases used during the parsing of the expression.

   LongDouble_t*        fConstLD;   //! local version of fConsts able to store bigger numbers

   RealInstanceCache fRealInstanceCache; //! Cache accelerating the GetRealInstance function

   TTreeFormula(const char *name, const char *formula, TTree *tree, const std::vector<std::string>& aliases);
   void Init(const char *name, const char *formula);
   Bool_t      BranchHasMethod(TLeaf* leaf, TBranch* branch, const char* method,const char* params, Long64_t readentry) const;
   Int_t       DefineAlternate(const char* expression);
   void        DefineDimensions(Int_t code, Int_t size, TFormLeafInfoMultiVarDim * info, Int_t& virt_dim);
   Int_t       FindLeafForExpression(const char* expression, TLeaf *&leaf, TString &leftover, Bool_t &final, UInt_t &paran_level, TObjArray &castqueue, std::vector<std::string>& aliasUsed, Bool_t &useLeafCollectionObject, const char *fullExpression);
   TLeaf*      GetLeafWithDatamember(const char* topchoice, const char* nextchice, Long64_t readentry) const;
   Int_t       ParseWithLeaf(TLeaf *leaf, const char *expression, Bool_t final, UInt_t paran_level, TObjArray &castqueue, Bool_t useLeafCollectionObject, const char *fullExpression);
   Int_t       RegisterDimensions(Int_t code, Int_t size, TFormLeafInfoMultiVarDim * multidim = 0);
   Int_t       RegisterDimensions(Int_t code, TBranchElement *branch);
   Int_t       RegisterDimensions(Int_t code, TFormLeafInfo *info, TFormLeafInfo *maininfo, Bool_t useCollectionObject);
   Int_t       RegisterDimensions(Int_t code, TLeaf *leaf);
   Int_t       RegisterDimensions(const char *size, Int_t code);

   virtual Double_t  GetValueFromMethod(Int_t i, TLeaf *leaf) const;
   virtual void*     GetValuePointerFromMethod(Int_t i, TLeaf *leaf) const;
   Int_t             GetRealInstance(Int_t instance, Int_t codeindex);

   void              LoadBranches();
   Bool_t            LoadCurrentDim();
   void              ResetDimensions();

   virtual TClass*   EvalClass(Int_t oper) const;
   virtual Bool_t    IsLeafInteger(Int_t code) const;
   virtual Bool_t    IsString(Int_t oper) const;
   virtual Bool_t    IsLeafString(Int_t code) const;
   virtual Bool_t    SwitchToFormLeafInfo(Int_t code);
   virtual Bool_t    StringToNumber(Int_t code);

   void              Convert(UInt_t fromVersion);

private:
   // Not implemented yet
   TTreeFormula(const TTreeFormula&);
   TTreeFormula& operator=(const TTreeFormula&);

   template<typename T> T GetConstant(Int_t k);

public:
   TTreeFormula();
   TTreeFormula(const char *name,const char *formula, TTree *tree);
   virtual   ~TTreeFormula();

   virtual Int_t       DefinedVariable(TString &variable, Int_t &action);
   virtual TClass*     EvalClass() const;

   template<typename T> T EvalInstance(Int_t i=0, const char *stringStack[]=0);
   virtual Double_t       EvalInstance(Int_t i=0, const char *stringStack[]=0) {return EvalInstance<Double_t>(i, stringStack); }
   virtual Long64_t       EvalInstance64(Int_t i=0, const char *stringStack[]=0) {return EvalInstance<Long64_t>(i, stringStack); }
   virtual LongDouble_t   EvalInstanceLD(Int_t i=0, const char *stringStack[]=0) {return EvalInstance<LongDouble_t>(i, stringStack); }

   virtual const char *EvalStringInstance(Int_t i=0);
   virtual void*       EvalObject(Int_t i=0);
   // EvalInstance should be const.  See comment on GetNdata()
   TFormLeafInfo      *GetLeafInfo(Int_t code) const;
   TTreeFormulaManager*GetManager() const { return fManager; }
   TMethodCall        *GetMethodCall(Int_t code) const;
   virtual Int_t       GetMultiplicity() const {return fMultiplicity;}
   virtual TLeaf      *GetLeaf(Int_t n) const;
   virtual Int_t       GetNcodes() const {return fNcodes;}
   virtual Int_t       GetNdata();
   //GetNdata should probably be const.  However it need to cache some information about the actual dimension
   //of arrays, so if GetNdata is const, the variables fUsedSizes and fCumulUsedSizes need to be declared
   //mutable.  We will be able to do that only when all the compilers supported for ROOT actually implemented
   //the mutable keyword.
   //NOTE: Also modify the code in PrintValue which current goes around this limitation :(
   virtual Bool_t      IsInteger(Bool_t fast=kTRUE) const;
           Bool_t      IsQuickLoad() const { return fQuickLoad; }
   virtual Bool_t      IsString() const;
   virtual Bool_t      Notify() { UpdateFormulaLeaves(); return kTRUE; }
   virtual char       *PrintValue(Int_t mode=0) const;
   virtual char       *PrintValue(Int_t mode, Int_t instance, const char *decform = "9.9") const;
   virtual void        SetAxis(TAxis *axis=0);
           void        SetQuickLoad(Bool_t quick) { fQuickLoad = quick; }
   virtual void        SetTree(TTree *tree) {fTree = tree;}
   virtual void        ResetLoading();
   virtual TTree*      GetTree() const {return fTree;}
   virtual void        UpdateFormulaLeaves();

   ClassDef(TTreeFormula, 10);  //The Tree formula
};

#endif
