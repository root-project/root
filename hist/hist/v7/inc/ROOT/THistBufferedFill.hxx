/// \file ROOT/THistBufferedFill.h
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

#ifndef ROOT7_THistBufferedFill
#define ROOT7_THistBufferedFill

#include "ROOT/RArrayView.hxx"

namespace ROOT {
namespace Experimental {

namespace Internal {
template <class DERIVED, class HIST, int SIZE>
class THistBufferedFillBase {
public:
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

private:
   size_t fCursor = 0;
   std::array<CoordArray_t, SIZE> fXBuf;
   std::array<Weight_t, SIZE> fWBuf;

public:
   THistBufferedFillBase() {}
   ~THistBufferedFillBase() { toDerived().Flush(); }

   DERIVED &toDerived() { return *static_cast<DERIVED *>(this); }
   const DERIVED &toDerived() const { return *static_cast<const DERIVED *>(this); }

   std::array_view<CoordArray_t> GetCoords() const
   {
      return std::array_view<CoordArray_t>(fXBuf.begin(), fXBuf.begin() + fCursor);
   }
   std::array_view<Weight_t> GetWeights() const
   {
      return std::array_view<Weight_t>(fWBuf.begin(), fWBuf.begin() + fCursor);
   }

   void Fill(const CoordArray_t &x, Weight_t weight = 1.)
   {
      fXBuf[fCursor] = x;
      fWBuf[fCursor++] = weight;
      if (fCursor == SIZE) {
         toDerived().Flush();
         fCursor = 0;
      }
   }
};

} // namespace Internal

/** \class THistBufferedFill
 Buffers calls to Fill().

 Once the buffer is full, on destruction of when calling Flush(), it sends the
 buffers off as an ideally vectorizable FillN() operation. It also serves as a
 multi-threaded way of filling the same histogram, reducing the locking
 frequency.

 The HIST template can be either a THist instance, a THistImpl instance, or
 a THistLockedFill instance.
 **/

template <class HIST, int SIZE = 1024>
class THistBufferedFill: public Internal::THistBufferedFillBase<THistBufferedFill<HIST, SIZE>, HIST, SIZE> {
public:
   using Hist_t = HIST;
   using CoordArray_t = typename HIST::CoordArray_t;
   using Weight_t = typename HIST::Weight_t;

private:
   HIST &fHist;
   size_t fCursor = 0;
   std::array<CoordArray_t, SIZE> fXBuf;
   std::array<Weight_t, SIZE> fWBuf;

public:
   THistBufferedFill(Hist_t &hist): fHist{hist} {}

   void FillN(const std::array_view<CoordArray_t> xN, const std::array_view<Weight_t> weightN)
   {
      fHist.FillN(xN, weightN);
   }

   void FillN(const std::array_view<CoordArray_t> xN) { fHist.FillN(xN); }

   void Flush() { fHist.FillN(this->GetCoords(), this->GetWeights()); }

   HIST &GetHist()
   {
      Flush(); // synchronize!
      return fHist;
   }
   operator HIST &() { return GetHist(); }

   static constexpr int GetNDim() { return HIST::GetNDim(); }
};
} // namespace Experimental
} // namespace ROOT

#endif
