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


#include "TVirtualIndex.h"

#include <vector>
#include <utility>

class TTreeFormula;
class TTreeIndex;
class TChain;

class TChainIndex : public TVirtualIndex {
public:
   // holds a description of indices of trees in the chain.
   class TChainIndexEntry {
      void Swap(TChainIndexEntry &other);

   public:
      TChainIndexEntry() : fMinIndexValue(0), fMinIndexValMinor(0),
                           fMaxIndexValue(0), fMaxIndexValMinor(0),
                           fTreeIndex(nullptr) {}
      TChainIndexEntry(const TChainIndexEntry &other);
      TChainIndexEntry &operator=(TChainIndexEntry other)
      {
         other.Swap(*this);
         return *this;
      }
      typedef std::pair<Long64_t, Long64_t>      IndexValPair_t;

      IndexValPair_t GetMinIndexValPair() const { return IndexValPair_t(fMinIndexValue, fMinIndexValMinor); }
      IndexValPair_t GetMaxIndexValPair() const { return IndexValPair_t(fMaxIndexValue, fMaxIndexValMinor); }
      void           SetMinMaxFrom(const TTreeIndex *index );

      Long64_t    fMinIndexValue;           // the minimum value of the index (upper bits)
      Long64_t    fMinIndexValMinor;        // the minimum value of the index (lower bits)
      Long64_t    fMaxIndexValue;           // the maximum value of the index (upper bits)
      Long64_t    fMaxIndexValMinor;        // the maximum value of the index (lower bits)
      TVirtualIndex* fTreeIndex;            // the tree index in case it was created in the constructor,
                                            // otherwise 0
   };
protected:

   TString        fMajorName;               // Index major name
   TString        fMinorName;               // Index minor name
   TTreeFormula  *fMajorFormulaParent;      //! Pointer to major TreeFormula in Parent tree (if any)
   TTreeFormula  *fMinorFormulaParent;      //! Pointer to minor TreeFormula in Parent tree (if any)
   std::vector<TChainIndexEntry> fEntries;  // descriptions of indices of trees in the chain.

   std::pair<TVirtualIndex*, Int_t> GetSubTreeIndex(Long64_t major, Long64_t minor) const;
   void ReleaseSubTreeIndex(TVirtualIndex* index, Int_t treeNo) const;
   void DeleteIndices();

   TTreeFormula  *GetMajorFormulaParent(const TTree *parent);
   TTreeFormula  *GetMinorFormulaParent(const TTree *parent);

public:
   TChainIndex();
   TChainIndex(const TTree *T, const char *majorname, const char *minorname, bool long64major = false, bool long64minor = false);
   ~TChainIndex() override;
   void           Append(const TVirtualIndex *, bool delaySort = false) override;
   Long64_t       GetEntryNumberFriend(const TTree *parent) override;
   Long64_t       GetEntryNumberWithIndex(Long64_t major, Long64_t minor) const override;
   Long64_t       GetEntryNumberWithBestIndex(Long64_t major, Long64_t minor) const override;
   const char    *GetMajorName()    const override {return fMajorName.Data();}
   const char    *GetMinorName()    const override {return fMinorName.Data();}
   Long64_t       GetN()            const override {return fEntries.size();}
   bool           IsValidFor(const TTree *parent) override;
   void           UpdateFormulaLeaves(const TTree *parent) override;
   void           SetTree(TTree *T) override;
   TObject *Clone(const char *newname = "") const override;

   ClassDefOverride(TChainIndex,1)  //A Tree Index with majorname and minorname.
};

#endif
