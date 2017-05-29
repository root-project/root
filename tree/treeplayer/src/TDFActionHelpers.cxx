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

CountHelper::CountHelper(const std::shared_ptr<unsigned int> &resultCount, unsigned int nSlots)
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

void FillHelper::UpdateMinMax(unsigned int slot, double v)
{
   auto &thisMin = fMin[slot];
   auto &thisMax = fMax[slot];
   thisMin = std::min(thisMin, v);
   thisMax = std::max(thisMax, v);
}

FillHelper::FillHelper(const std::shared_ptr<Hist_t> &h, unsigned int nSlots)
   : fResultHist(h), fNSlots(nSlots), fBufSize(fgTotalBufSize / nSlots),
     fMin(nSlots, std::numeric_limits<BufEl_t>::max()), fMax(nSlots, std::numeric_limits<BufEl_t>::min())
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
       globalMax != std::numeric_limits<BufEl_t>::min()) {
      fResultHist->SetBins(fResultHist->GetNbinsX(), globalMin, globalMax);
   }

   for (unsigned int i = 0; i < fNSlots; ++i) {
      // TODO: Here one really needs to fix FillN!
      if (fWBuffers[i].empty()) {
         fWBuffers[i].resize(fBuffers[i].size(), 1.);
      }
      fResultHist->FillN(fBuffers[i].size(), fBuffers[i].data(), fWBuffers[i].data());
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

MinHelper::MinHelper(const std::shared_ptr<double> &minVPtr, unsigned int nSlots)
   : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<double>::max())
{
}

void MinHelper::Exec(unsigned int slot, double v)
{
   fMins[slot] = std::min(v, fMins[slot]);
}

void MinHelper::Finalize()
{
   *fResultMin = std::numeric_limits<double>::max();
   for (auto &m : fMins) *fResultMin = std::min(m, *fResultMin);
}

template void MinHelper::Exec(unsigned int, const std::vector<float> &);
template void MinHelper::Exec(unsigned int, const std::vector<double> &);
template void MinHelper::Exec(unsigned int, const std::vector<char> &);
template void MinHelper::Exec(unsigned int, const std::vector<int> &);
template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

MaxHelper::MaxHelper(const std::shared_ptr<double> &maxVPtr, unsigned int nSlots)
   : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<double>::min())
{
}

void MaxHelper::Exec(unsigned int slot, double v)
{
   fMaxs[slot] = std::max(v, fMaxs[slot]);
}

void MaxHelper::Finalize()
{
   *fResultMax = std::numeric_limits<double>::min();
   for (auto &m : fMaxs) {
      *fResultMax = std::max(m, *fResultMax);
   }
}

template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

MeanHelper::MeanHelper(const std::shared_ptr<double> &meanVPtr, unsigned int nSlots)
   : fResultMean(meanVPtr), fCounts(nSlots, 0), fSums(nSlots, 0)
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
   Count_t sumOfCounts = 0;
   for (auto &c : fCounts) sumOfCounts += c;
   *fResultMean = sumOfSums / (sumOfCounts > 0 ? sumOfCounts : 1);
}

template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

} // end NS TDF
} // end NS Internal
} // end NS ROOT
