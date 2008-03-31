// @(#)root/treeplayer:$Id$
// Author: Marek Biskup  07/06/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChainIndex
#define ROOT_TChainIndex


//////////////////////////////////////////////////////////////////////////
//
// TChainIndex
//
// A Chain Index with majorname and minorname.
// It uses tree indices of all the trees in the chain instead of building
// a new index.
// The index values from the first tree should be less then 
// all the index values from the second tree, and so on.
// If a tree in the chain doesn't have an index the index will be created
// and kept inside this chain index.
// 
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualIndex
#include "TVirtualIndex.h"
#endif

#include <vector>

class TTreeFormula;
class TChain;

class TChainIndex : public TVirtualIndex {

public:
   class TChainIndexEntry {
      // holds a description of indices of trees in the chain. 
   public:
      TChainIndexEntry() : fMinIndexValue(0), fMaxIndexValue(0), fTreeIndex(0) {}

      Long64_t    fMinIndexValue;           // the minimum value of the index
      Long64_t    fMaxIndexValue;           // the maximum value of the index
      TVirtualIndex* fTreeIndex;            // the tree index in case it was created in the constructor,
                                            // otherwise 0
   };
protected:

   TString        fMajorName;               // Index major name
   TString        fMinorName;               // Index minor name
   TTreeFormula  *fMajorFormulaParent;      //! Pointer to major TreeFormula in Parent tree (if any)
   TTreeFormula  *fMinorFormulaParent;      //! Pointer to minor TreeFormula in Parent tree (if any)
   std::vector<TChainIndexEntry> fEntries; // descriptions of indices of trees in the chain.

   std::pair<TVirtualIndex*, Int_t> GetSubTreeIndex(Int_t major, Int_t minor) const;
   void ReleaseSubTreeIndex(TVirtualIndex* index, Int_t treeNo) const;
   void DeleteIndices();

public:
   TChainIndex();
   TChainIndex(const TTree *T, const char *majorname, const char *minorname);
   virtual               ~TChainIndex();
   virtual void           Append(const TVirtualIndex *, Bool_t delaySort = kFALSE);
   virtual Int_t          GetEntryNumberFriend(const TTree *T);
   virtual Long64_t       GetEntryNumberWithIndex(Int_t major, Int_t minor) const;
   virtual Long64_t       GetEntryNumberWithBestIndex(Int_t major, Int_t minor) const;
   const char            *GetMajorName()    const {return fMajorName.Data();}
   const char            *GetMinorName()    const {return fMinorName.Data();}
   virtual Long64_t       GetN()            const {return fEntries.size();}
   virtual TTreeFormula  *GetMajorFormulaParent(const TTree *T);
   virtual TTreeFormula  *GetMinorFormulaParent(const TTree *T);
   virtual void           UpdateFormulaLeaves(const TTree *parent);
   virtual void           SetTree(const TTree *T);

   ClassDef(TChainIndex,1)  //A Tree Index with majorname and minorname.
};

#endif

