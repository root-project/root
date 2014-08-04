// @(#)root/treeplayer:$Id$
// Author: Philippe Canal   20/03/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeFormulaManager
#define ROOT_TTreeFormulaManager


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFormulaManager                                                  //
//                                                                      //
// A class coordinating several TTreeFormula objects.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#include "TTreeFormula.h"

class TArrayI;


class TTreeFormulaManager : public TObject {
private:
   TObjArray   fFormulas;
   Int_t       fMultiplicity;     // Indicator of the variability of the formula
   Bool_t      fMultiVarDim;      // True if one of the variable has 2 variable size dimensions.
   Int_t       fNdata;            //! Last value calculated by GetNdata

   //the next line should be: mutable Int_t fCumulUsedSizes[kMAXFORMDIM+1]; See GetNdata()
   Int_t       fCumulUsedSizes[kMAXFORMDIM+1];      //Accumulated size of lower dimensions as seen for this entry
   TArrayI    *fCumulUsedVarDims;                   //fCumulUsedSizes(1) for multi variable dimensions case
   //the next line should be: mutable Int_t fUsedSizes[kMAXFORMDIM+1]; See GetNdata()
   Int_t       fUsedSizes[kMAXFORMDIM+1];           //Actual size of the dimensions as seen for this entry.
   TArrayI    *fVarDims[kMAXFORMDIM+1];             //List of variable sizes dimensions.
   Int_t       fVirtUsedSizes[kMAXFORMDIM+1];       //Virtual size of lower dimensions as seen for this formula

   Bool_t      fNeedSync;         // Indicate whether a new formula has been added since the last synchronization

   friend class TTreeFormula;

private:
   // Not implemented yet
   TTreeFormulaManager(const TTreeFormulaManager&);
   TTreeFormulaManager& operator=(const TTreeFormulaManager&);

#if !defined(_MSC_VER) || (_MSC_VER < 1300) || defined(__CINT__)
   // Deletion is the sole responsability of the TTreeFormulas
   ~TTreeFormulaManager();
#endif

protected:

   virtual void       AddVarDims(Int_t virt_dim);
   virtual void       CancelDimension(Int_t virt_dim);
   virtual void       EnableMultiVarDims();
   virtual void       UpdateUsedSize(Int_t &virt_dim, Int_t vsize);

public:
   TTreeFormulaManager();
#if _MSC_VER >= 1300
   // cannot be private due to a bug in VC++7
   ~TTreeFormulaManager();
#endif

   virtual void       Add(TTreeFormula*);
   virtual Int_t      GetMultiplicity() const {return fMultiplicity;}
   virtual Int_t      GetNdata(Bool_t forceLoadDim = kFALSE);
   virtual Bool_t     Notify() { UpdateFormulaLeaves(); return kTRUE; }
   virtual void       Remove(TTreeFormula*);
   virtual Bool_t     Sync();
   virtual void       UpdateFormulaLeaves();

   ClassDef(TTreeFormulaManager,0) // A class coordinating several TTreeFormula objects.
};


#endif // ROOT_TTreeFormulaManager

