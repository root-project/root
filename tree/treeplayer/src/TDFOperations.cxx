// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include "ROOT/TDFOperations.hxx"

namespace ROOT {

namespace Internal {

namespace Operations {

CountOperation::CountOperation(unsigned int *resultCount, unsigned int nSlots) : fResultCount(resultCount), fCounts(nSlots, 0) {}

void CountOperation::Exec(unsigned int slot)
{
   fCounts[slot]++;
}

void CountOperation::Finalize()
{
   *fResultCount = 0;
   for (auto &c : fCounts) {
      *fResultCount += c;
   }
}

CountOperation::~CountOperation()
{
   Finalize();
}

void FillOperation::UpdateMinMax(unsigned int slot, double v) {
   auto& thisMin = fMin[slot];
   auto& thisMax = fMax[slot];
   thisMin = std::min(thisMin, v);
   thisMax = std::max(thisMax, v);
}


FillOperation::FillOperation(std::shared_ptr<Hist_t> h, unsigned int nSlots) : fResultHist(h),
                                                                               fNSlots(nSlots),
                                                                               fBufSize (fgTotalBufSize / nSlots),
                                                                               fMin(nSlots, std::numeric_limits<BufEl_t>::max()),
                                                                               fMax(nSlots, std::numeric_limits<BufEl_t>::min())
{
   fBuffers.reserve(fNSlots);
   fWBuffers.reserve(fNSlots);
   for (unsigned int i=0; i<fNSlots; ++i) {
      Buf_t v;
      v.reserve(fBufSize);
      fBuffers.emplace_back(v);

      Buf_t w(fBufSize,1);
      fWBuffers.emplace_back(v);
   }
}

void FillOperation::Exec(double v, unsigned int slot)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
}

void FillOperation::Exec(double v, double w, unsigned int slot)
{
   UpdateMinMax(slot, v);
   fBuffers[slot].emplace_back(v);
   fWBuffers[slot].emplace_back(w);
}

void FillOperation::Finalize()
{
   for (unsigned int i=0; i<fNSlots; ++i) {
      if (fBuffers[i].size() != fBuffers[i].size()) {
         throw std::runtime_error("Cannot fill weighted histogram with values in containers of different sizes.");
      }
   }

   BufEl_t globalMin = *std::min_element(fMin.begin(), fMin.end());
   BufEl_t globalMax = *std::max_element(fMax.begin(), fMax.end());

   if (fResultHist->CanExtendAllAxes() &&
       globalMin != std::numeric_limits<BufEl_t>::max() &&
       globalMax != std::numeric_limits<BufEl_t>::min()) {
      auto xaxis = fResultHist->GetXaxis();
      fResultHist->ExtendAxis(globalMin, xaxis);
      fResultHist->ExtendAxis(globalMax, xaxis);
   }

   for (unsigned int i=0; i<fNSlots; ++i) {
      // TODO: Here one really needs to fix FillN!
      if (fWBuffers[i].empty()) {
         fWBuffers[i].resize(fBuffers[i].size(), 1.);
      }
      fResultHist->FillN(fBuffers[i].size(), fBuffers[i].data(),  fWBuffers[i].data());
   }
}

FillOperation::~FillOperation()
{
   Finalize();
}

template void FillOperation::Exec(const std::vector<float>&, unsigned int);
template void FillOperation::Exec(const std::vector<double>&, unsigned int);
template void FillOperation::Exec(const std::vector<char>&, unsigned int);
template void FillOperation::Exec(const std::vector<int>&, unsigned int);
template void FillOperation::Exec(const std::vector<unsigned int>&, unsigned int);
template void FillOperation::Exec(const std::vector<float>&, const std::vector<float>&, unsigned int);
template void FillOperation::Exec(const std::vector<double>&, const std::vector<double>&, unsigned int);
template void FillOperation::Exec(const std::vector<char>&, const std::vector<char>&, unsigned int);
template void FillOperation::Exec(const std::vector<int>&, const std::vector<int>&, unsigned int);
template void FillOperation::Exec(const std::vector<unsigned int>&, const std::vector<unsigned int>&, unsigned int);

MinOperation::MinOperation(double *minVPtr, unsigned int nSlots)
   : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<double>::max()) { }

void MinOperation::Exec(double v, unsigned int slot)
{
   fMins[slot] = std::min(v, fMins[slot]);
}

void MinOperation::Finalize()
{
   *fResultMin = std::numeric_limits<double>::max();
   for (auto &m : fMins) *fResultMin = std::min(m, *fResultMin);
}

MinOperation::~MinOperation()
{
   Finalize();
}

template void MinOperation::Exec(const std::vector<float>&, unsigned int);
template void MinOperation::Exec(const std::vector<double>&, unsigned int);
template void MinOperation::Exec(const std::vector<char>&, unsigned int);
template void MinOperation::Exec(const std::vector<int>&, unsigned int);
template void MinOperation::Exec(const std::vector<unsigned int>&, unsigned int);


MaxOperation::MaxOperation(double *maxVPtr, unsigned int nSlots)
   : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<double>::min()) { }

void MaxOperation::Exec(double v, unsigned int slot)
{
   fMaxs[slot] = std::max(v, fMaxs[slot]);
}

void MaxOperation::Finalize()
{
   *fResultMax = std::numeric_limits<double>::min();
   for (auto &m : fMaxs) {
      *fResultMax = std::max(m, *fResultMax);
   }
}

MaxOperation::~MaxOperation()
{
   Finalize();
}

template void MaxOperation::Exec(const std::vector<float>&, unsigned int);
template void MaxOperation::Exec(const std::vector<double>&, unsigned int);
template void MaxOperation::Exec(const std::vector<char>&, unsigned int);
template void MaxOperation::Exec(const std::vector<int>&, unsigned int);
template void MaxOperation::Exec(const std::vector<unsigned int>&, unsigned int);


MeanOperation::MeanOperation(double *meanVPtr, unsigned int nSlots) : fResultMean(meanVPtr), fCounts(nSlots, 0), fSums(nSlots, 0) {}

void MeanOperation::Exec(double v, unsigned int slot)
{
   fSums[slot] += v;
   fCounts[slot] ++;
}

void MeanOperation::Finalize()
{
   double sumOfSums = 0;
   for (auto &s : fSums) sumOfSums += s;
   Count_t sumOfCounts = 0;
   for (auto &c : fCounts) sumOfCounts += c;
   *fResultMean = sumOfSums / (sumOfCounts > 0 ? sumOfCounts : 1);
}

MeanOperation::~MeanOperation()
{
   Finalize();
}

template void MeanOperation::Exec(const std::vector<float>&, unsigned int);
template void MeanOperation::Exec(const std::vector<double>&, unsigned int);
template void MeanOperation::Exec(const std::vector<char>&, unsigned int);
template void MeanOperation::Exec(const std::vector<int>&, unsigned int);
template void MeanOperation::Exec(const std::vector<unsigned int>&, unsigned int);


} // end NS Operations
} // end NS Internal
} // end NS ROOT