// @(#)root/treeplayer:$Name:  $:$Id: TTreeFormula.h,v 1.11 2001/04/20 21:21:38 brun Exp $
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

#ifndef ROOT_TFormula
#include "TFormula.h"
#endif

#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

const Int_t kMAXCODES = kMAXFOUND; // must be the same as kMAXFOUND in TFormula
const Int_t kMAXFORMDIM = 5; // Maximum number of array dimensions support in TTreeFormula

class TTree;
class TMethodCall;
class TLeafObject;
class TDataMember;
class TStreamerElement;
class TFormLeafInfo;
class TTreeFormula : public TFormula {

protected:
   enum { kIsCharacter = BIT(12) };
   enum { kDirect, kDataMember, kMethod };

   TTree       *fTree;            //! pointer to Tree
   Short_t     fCodes[kMAXCODES]; //  List of leaf numbers referenced in formula
   Int_t       fNdata[kMAXCODES]; //! This caches the physical number of element in the leaf or datamember.
   Int_t       fNcodes;           //  Number of leaves referenced in formula
   Int_t       fMultiplicity;     //  Number of array elements in leaves in case of a TClonesArray
   Int_t       fInstance;         //  Instance number for GetValue
   Int_t       fNindex;           //  Size of fIndex
   Int_t      *fLookupType;       //[fNindex] array indicating how each leaf should be looked-up
   TObjArray   fLeaves;           //!  List of leaf used in this formula.
   TObjArray   fDataMembers;      //!  List of leaf data members
   TObjArray   fMethods;          //!  List of leaf method calls
   TObjArray   fNames;            //  List of TNamed describing leaves
   
   Int_t         fNdimensions[kMAXCODES];             //Number of array dimensions in each leaf
   Int_t         fCumulSizes[kMAXCODES][kMAXFORMDIM]; //Accumulated size of lower dimensions for each leaf
   //mutable Int_t fUsedSizes[kMAXFORMDIM+1]; See GetNdata()
   Int_t fUsedSizes[kMAXFORMDIM+1]; //Actual size of the dimensions as seen for this entry.
   //mutable Int_t fCumulUsedSizes[kMAXFORMDIM+1]; See GetNdata()
   Int_t fCumulUsedSizes[kMAXFORMDIM+1]; //Accumulated size of lower dimensions as seen for this entry.
   Int_t         fVirtUsedSizes[kMAXFORMDIM+1];       //Virtual size of lower dimensions as seen for this formula
   Int_t         fIndexes[kMAXCODES][kMAXFORMDIM];    //Index of array selected by user for each leaf
   TTreeFormula *fVarIndexes[kMAXCODES][kMAXFORMDIM]; //Pointer to a variable index.

   void        DefineDimensions(Int_t code, TFormLeafInfo *info,  Int_t& virt_dim);
   void        DefineDimensions(const char *size, Int_t code, Int_t& virt_dim);
   virtual Double_t   GetValueFromMethod(Int_t i, TLeaf *leaf) const;
public:
             TTreeFormula();
             TTreeFormula(const char *name,const char *formula, TTree *tree);
   virtual   ~TTreeFormula();
   virtual Int_t      DefinedVariable(TString &variable);
   virtual Double_t   EvalInstance(Int_t i=0) const;
   TObject           *GetLeafInfo(Int_t code) const;
   TMethodCall       *GetMethodCall(Int_t code) const;
   virtual Int_t      GetMultiplicity() const {return fMultiplicity;}
   virtual TLeaf     *GetLeaf(Int_t n) const;
   virtual Int_t      GetNcodes() const {return fNcodes;}
   virtual Int_t      GetNdata();
   //GetNdata should probably be const.  However it need to cache some information about the actual dimension
   //of arrays, so if GetNdata is const, the variables fUsedSizes and fCumulUsedSizes need to be declared
   //mutable.  We will be able to do that only when all the compilers supported for ROOT actually implemented
   //the mutable keyword. 
   //NOTE: Also modify the code in PrintValue which current goes around this limitation :(
   virtual char      *PrintValue(Int_t mode=0) const;
   virtual void       SetTree(TTree *tree) {fTree = tree;}
   virtual void       UpdateFormulaLeaves();

   ClassDef(TTreeFormula,4)  //The Tree formula
};

#endif
