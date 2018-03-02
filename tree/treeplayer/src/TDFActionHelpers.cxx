// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDFActionHelpers.hxx"

namespace ROOT {
namespace Internal {
namespace TDF {

CountHelper::CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots)
   : fResultCount(resultCount), fCounts(nSlots, 0)
{
}

void CountHelper::Exec(unsigned int slot)
{
   fCounts[slot]++;
}

void CountHelper::Finalize()
{
   *fResultCount = 0;
   for (auto &c : fCounts) {
      *fResultCount += c;
   }
}

ULong64_t &CountHelper::PartialUpdate(unsigned int slot)
{
   return fCounts[slot];
}

void FillHelper::UpdateMinMax(unsigned int slot, double v)
{
   auto &thisMin = fMin[slot];
   auto &thisMax = fMax[slot];
   thisMin = std::min(thisMin, v);
   thisMax = std::max(thisMax, v);
}

FillHelper::FillHelper(const std::shared_ptr<Hist_t> &h, const unsigned int nSlots)
   : fResultHist(h), fNSlots(nSlots), fBufSize(fgTotalBufSize / nSlots), fPartialHists(fNSlots),
     fMin(nSlots, std::numeric_limits<BufEl_t>::max()), fMax(nSlots, std::numeric_limits<BufEl_t>::lowest())
{
   fBuffers.reserve(fNSlots);
   fWBuffers.reserve(fNSlots);
   for (unsigned int i = 0; i < fNSlots; ++i) {
      Buf_t v;
      v.reserve(fBufSize);
      fBuffers.emplace_back(v);
      fWBuffers.emplace_back(v);
   }
}

void FillHelper::Exec(unsigned int slot, double v)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
}

void FillHelper::Exec(unsigned int slot, double v, double w)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
   fWBuffers[slot].emplace_back(w);
}

Hist_t &FillHelper::PartialUpdate(unsigned int slot)
{
   auto &partialHist = fPartialHists[slot];
   // TODO it is inefficient to re-create the partial histogram everytime the callback is called
   //      ideally we could incrementally fill it with the latest entries in the buffers
   partialHist.reset(new Hist_t(*fResultHist));
   auto weights = fWBuffers[slot].empty() ? nullptr : fWBuffers[slot].data();
   partialHist->FillN(fBuffers[slot].size(), fBuffers[slot].data(), weights);
   return *partialHist;
}

void FillHelper::Finalize()
{
   for (unsigned int i = 0; i < fNSlots; ++i) {
      if (!fWBuffers[i].empty() && fBuffers[i].size() != fWBuffers[i].size()) {
         throw std::runtime_error("Cannot fill weighted histogram with values in containers of different sizes.");
      }
   }

   BufEl_t globalMin = *std::min_element(fMin.begin(), fMin.end());
   BufEl_t globalMax = *std::max_element(fMax.begin(), fMax.end());

   if (fResultHist->CanExtendAllAxes() && globalMin != std::numeric_limits<BufEl_t>::max() &&
       globalMax != std::numeric_limits<BufEl_t>::lowest()) {
      fResultHist->SetBins(fResultHist->GetNbinsX(), globalMin, globalMax);
   }

   for (unsigned int i = 0; i < fNSlots; ++i) {
      auto weights = fWBuffers[i].empty() ? nullptr : fWBuffers[i].data();
      fResultHist->FillN(fBuffers[i].size(), fBuffers[i].data(), weights);
   }
}

template void FillHelper::Exec(unsigned int, const std::vector<float> &);
template void FillHelper::Exec(unsigned int, const std::vector<double> &);
template void FillHelper::Exec(unsigned int, const std::vector<char> &);
template void FillHelper::Exec(unsigned int, const std::vector<int> &);
template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &);
template void FillHelper::Exec(unsigned int, const std::vector<float> &, const std::vector<float> &);
template void FillHelper::Exec(unsigned int, const std::vector<double> &, const std::vector<double> &);
template void FillHelper::Exec(unsigned int, const std::vector<char> &, const std::vector<char> &);
template void FillHelper::Exec(unsigned int, const std::vector<int> &, const std::vector<int> &);
template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &, const std::vector<unsigned int> &);

// TODO
// template void MinHelper::Exec(unsigned int, const std::vector<float> &);
// template void MinHelper::Exec(unsigned int, const std::vector<double> &);
// template void MinHelper::Exec(unsigned int, const std::vector<char> &);
// template void MinHelper::Exec(unsigned int, const std::vector<int> &);
// template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

// template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
// template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
// template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
// template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
// template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

MeanHelper::MeanHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots)
   : fResultMean(meanVPtr), fCounts(nSlots, 0), fSums(nSlots, 0), fPartialMeans(nSlots)
{
}

void MeanHelper::Exec(unsigned int slot, double v)
{
   fSums[slot] += v;
   fCounts[slot]++;
}

void MeanHelper::Finalize()
{
   double sumOfSums = 0;
   for (auto &s : fSums) sumOfSums += s;
   ULong64_t sumOfCounts = 0;
   for (auto &c : fCounts) sumOfCounts += c;
   *fResultMean = sumOfSums / (sumOfCounts > 0 ? sumOfCounts : 1);
}

double &MeanHelper::PartialUpdate(unsigned int slot)
{
   fPartialMeans[slot] = fSums[slot] / fCounts[slot];
   return fPartialMeans[slot];
}

template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

} // end NS TDF
} // end NS Internal
} // end NS ROOT
