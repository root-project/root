// @(#)root/treeplayer:$Id$
// Author: Rene Brun   05/07/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeIndex
#define ROOT_TTreeIndex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeIndex                                                           //
//                                                                      //
// A Tree Index with majorname and minorname.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TVirtualIndex.h"

class TTreeFormula;

class TTreeIndex : public TVirtualIndex {
protected:
   TString        fMajorName;           ///< Index major name
   TString        fMinorName;           ///< Index minor name
   Long64_t       fN;                   ///< Number of entries
   Long64_t      *fIndexValues;         ///<[fN] Sorted index values, higher 64bits
   Long64_t      *fIndexValuesMinor;    ///<[fN] Sorted index values, lower 64bits
   Long64_t      *fIndex;               ///<[fN] Index of sorted values
   TTreeFormula  *fMajorFormula;        ///<! Pointer to major TreeFormula
   TTreeFormula  *fMinorFormula;        ///<! Pointer to minor TreeFormula
   TTreeFormula  *fMajorFormulaParent;  ///<! Pointer to major TreeFormula in Parent tree (if any)
   TTreeFormula  *fMinorFormulaParent;  ///<! Pointer to minor TreeFormula in Parent tree (if any)

   TTreeFormula  *GetMajorFormulaParent(const TTree *parent);
   TTreeFormula  *GetMinorFormulaParent(const TTree *parent);

private:
   TTreeIndex(const TTreeIndex&) = delete;            // Not implemented.
   TTreeIndex &operator=(const TTreeIndex&) = delete; // Not implemented.

public:
   TTreeIndex();
   TTreeIndex(const TTree *T, const char *majorname, const char *minorname, bool long64major = false, bool long64minor = false);
   ~TTreeIndex() override;
   void                   Append(const TVirtualIndex *,bool delaySort = false) override;
   bool                   ConvertOldToNew();
   Long64_t               FindValues(Long64_t major, Long64_t minor) const;
   Long64_t               GetEntryNumberFriend(const TTree *parent) override;
   Long64_t               GetEntryNumberWithIndex(Long64_t major, Long64_t minor) const override;
   Long64_t               GetEntryNumberWithBestIndex(Long64_t major, Long64_t minor) const override;
   virtual Long64_t      *GetIndex()        const {return fIndex;}
   virtual Long64_t      *GetIndexValues()  const {return fIndexValues;}
   virtual Long64_t      *GetIndexValuesMinor()  const;
   const char            *GetMajorName()    const override {return fMajorName.Data();}
   const char            *GetMinorName()    const override {return fMinorName.Data();}
   Long64_t               GetN()            const override {return fN;}
   virtual TTreeFormula  *GetMajorFormula();
   virtual TTreeFormula  *GetMinorFormula();
   bool           IsValidFor(const TTree *parent) override;
   void           Print(Option_t *option="") const override;
   void           UpdateFormulaLeaves(const TTree *parent) override;
   void           SetTree(TTree *T) override;
   TObject *Clone(const char *newname = "") const override;

   ClassDefOverride(TTreeIndex,2);  //A Tree Index with majorname and minorname.
};

#endif

