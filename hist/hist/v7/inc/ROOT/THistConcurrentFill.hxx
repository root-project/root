/// \file ROOT/THistConcurrentFill.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistConcurrentFill
#define ROOT7_THistConcurrentFill

#include "ROOT/RArrayView.hxx"
#include "ROOT/THistBufferedFill.hxx"

#include <mutex>

namespace ROOT {
namespace Experimental {

template <class HIST, int SIZE>
class THistConcurrentFillManager;

/**
 \class THistConcurrentFiller
 Buffers a thread's Fill calls and submits them to the
 THistConcurrentFillManager. Enables multi-threaded filling.
 **/

template <class HIST, int SIZE>
class THistConcurrentFiller: public Internal::THistBufferedFillBase<THistConcurrentFiller<HIST, SIZE>, HIST, SIZE> {
   THistConcurrentFillManager<HIST, SIZE> &fManager;

public:
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

   THistConcurrentFiller(THistConcurrentFillManager<HIST, SIZE> &manager): fManager(manager) {}

   /// Thread-specific HIST::Fill().
   using Internal::THistBufferedFillBase<THistConcurrentFiller<HIST, SIZE>, HIST, SIZE>::Fill;

   /// Thread-specific HIST::FillN().
   void FillN(const std::array_view<CoordArray_t> xN, const std::array_view<Weight_t> weightN)
   {
      fManager.FillN(xN, weightN);
   }

   /// Thread-specific HIST::FillN().
   void FillN(const std::array_view<CoordArray_t> xN) { fManager.FillN(xN); }

   /// The buffer is full, flush it out.
   void Flush() { fManager.FillN(this->GetCoords(), this->GetWeights()); }

   HIST &GetHist() { return fManager->GetHist(); }
   operator HIST &() { return GetHist(); }

   static constexpr int GetNDim() { return HIST::GetNDim(); }
};

/**
 \class THistConcurrentFillManager
 Manages the synchronization of calls to FillN().

 The HIST template can be a THist instance. This class hands out
 THistConcurrentFiller objects that can concurrently fill the histogram. They
 buffer calls to Fill() until the buffer is full, and then swap the buffer
 with that of the THistConcurrentFillManager. The manager than fills the
 histogram.
 **/

template <class HIST, int SIZE = 1024>
class THistConcurrentFillManager {
   friend class THistConcurrentFiller<HIST, SIZE>;

public:
   using Hist_t = HIST;
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

private:
   HIST &fHist;
   std::mutex fFillMutex; // should become a spin lock

public:
   THistConcurrentFillManager(HIST &hist): fHist(hist) {}

   THistConcurrentFiller<HIST, SIZE> MakeFiller() { return THistConcurrentFiller<HIST, SIZE>{*this}; }

   /// Thread-specific HIST::FillN().
   void FillN(const std::array_view<CoordArray_t> xN, const std::array_view<Weight_t> weightN)
   {
      std::lock_guard<std::mutex> lockGuard(fFillMutex);
      fHist.FillN(xN, weightN);
   }

   /// Thread-specific HIST::FillN().
   void FillN(const std::array_view<CoordArray_t> xN)
   {
      std::lock_guard<std::mutex> lockGuard(fFillMutex);
      fHist.FillN(xN);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
