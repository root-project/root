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


#ifndef ROOT_TVirtualIndex
#include "TVirtualIndex.h"
#endif
#ifndef ROOT_TTreeFormula
#include "TTreeFormula.h"
#endif

class TTreeIndex : public TVirtualIndex {

protected:
   TString        fMajorName;           // Index major name
   TString        fMinorName;           // Index minor name
   Long64_t       fN;                   // Number of entries
   Long64_t      *fIndexValues;         //[fN] Sorted index values
   Long64_t      *fIndex;               //[fN] Index of sorted values
   TTreeFormula  *fMajorFormula;        //! Pointer to major TreeFormula
   TTreeFormula  *fMinorFormula;        //! Pointer to minor TreeFormula
   TTreeFormula  *fMajorFormulaParent;  //! Pointer to major TreeFormula in Parent tree (if any)
   TTreeFormula  *fMinorFormulaParent;  //! Pointer to minor TreeFormula in Parent tree (if any)
   
public:
   TTreeIndex();
   TTreeIndex(const TTree *T, const char *majorname, const char *minorname);
   virtual               ~TTreeIndex();
   virtual void           Append(const TVirtualIndex *,Bool_t delaySort = kFALSE);
   virtual Int_t          GetEntryNumberFriend(const TTree *T);
   virtual Long64_t       GetEntryNumberWithIndex(Int_t major, Int_t minor) const;
   virtual Long64_t       GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const;
   virtual Long64_t      *GetIndexValues()  const {return fIndexValues;}
   virtual Long64_t      *GetIndex()        const {return fIndex;}
   const char            *GetMajorName()    const {return fMajorName.Data();}
   const char            *GetMinorName()    const {return fMinorName.Data();}
   virtual Long64_t       GetN()            const {return fN;}
   virtual TTreeFormula  *GetMajorFormula();
   virtual TTreeFormula  *GetMinorFormula();
   virtual TTreeFormula  *GetMajorFormulaParent(const TTree *T);
   virtual TTreeFormula  *GetMinorFormulaParent(const TTree *T);
   virtual void           Print(Option_t *option="") const;
   virtual void           UpdateFormulaLeaves(const TTree *parent);
   virtual void           SetTree(const TTree *T);
   
   ClassDef(TTreeIndex,1);  //A Tree Index with majorname and minorname.
};

#endif

