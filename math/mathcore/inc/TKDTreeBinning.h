// @(#)root/mathcore:$Id$
// Authors: B. Rabacal   11/2010

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TKDTreeBinning
//
//

#include <map>
#include <utility>

#ifndef ROOT_TKDTree
#include "TKDTree.h"
#endif

namespace ROOT {
   namespace Fit {
      class BinData;
   }
}

class TKDTreeBinning : public TObject {
   Double_t** fData;
   std::vector<Double_t> fBinMinEdges;
   std::vector<Double_t> fBinMaxEdges;
   TKDTreeID* fDataBins;
   UInt_t fNBins;
   UInt_t fDim;
   UInt_t fDataSize;
   std::vector<std::pair<Double_t, Double_t> > fDataThresholds;
   std::vector<std::vector<std::pair<Bool_t, Bool_t> > > fCheckedBinEdges;
   std::vector<std::map<Double_t, std::vector<UInt_t> > > fCommonBinEdges;
   Bool_t fIsSorted;
   Bool_t fIsSortedAsc;
   std::vector<UInt_t> fBinsContent;
   struct CompareAsc;
   friend struct CompareAsc;
   struct CompareDesc;
   friend struct CompareDesc;
   TKDTreeBinning(TKDTreeBinning& bins);           // Disallowed copy constructor
   TKDTreeBinning operator=(TKDTreeBinning& bins); // Disallowed assign operator
   void SetData(Double_t* data);
   void SetTreeData();
   void SetBinsEdges();
   void SetBinMinMaxEdges(Double_t* binEdges);
   void SetCommonBinEdges(Double_t* binEdges);
   void SetBinsContent();
   void ReadjustMinBinEdges(Double_t* binEdges);
   void ReadjustMaxBinEdges(Double_t* binEdges);

public:

   // flag bits
   enum {
      kAdjustBinEdges     = BIT(14)  // adjust bin edges to avoid overlapping with data
   };


   TKDTreeBinning(UInt_t dataSize, UInt_t dataDim, Double_t* data, UInt_t nBins = 100);
   ~TKDTreeBinning();
   void SetNBins(UInt_t bins);
   void SortBinsByDensity(Bool_t sortAsc = kTRUE);
   const Double_t* GetBinsMinEdges() const;
   const Double_t* GetBinsMaxEdges() const;
   std::pair<const Double_t*, const Double_t*> GetBinsEdges() const;
   std::pair<const Double_t*, const Double_t*> GetBinEdges(UInt_t bin) const;
   const Double_t* GetBinMinEdges(UInt_t bin) const;
   const Double_t* GetBinMaxEdges(UInt_t bin) const;
   UInt_t GetNBins() const;
   UInt_t GetDim() const;
   UInt_t GetBinContent(UInt_t bin) const;
   TKDTreeID* GetTree() const;
   const Double_t* GetDimData(UInt_t dim) const;
   Double_t GetDataMin(UInt_t dim) const;
   Double_t GetDataMax(UInt_t dim) const;
   Double_t GetBinDensity(UInt_t bin) const;
   Double_t GetBinVolume(UInt_t bin) const;
   const Double_t* GetOneDimBinEdges() const;
   const Double_t* GetBinCenter(UInt_t bin) const;
   const Double_t* GetBinWidth(UInt_t bin) const;
   UInt_t GetBinMaxDensity() const;
   UInt_t GetBinMinDensity() const;
   void FillBinData(ROOT::Fit::BinData & data) const;

   ClassDef(TKDTreeBinning, 1)

};
