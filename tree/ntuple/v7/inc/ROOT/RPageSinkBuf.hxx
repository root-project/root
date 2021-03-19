/// \file ROOT/RPageSinkBuf.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
/// \date 2021-03-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageSinkBuf
#define ROOT7_RPageSinkBuf

#include <ROOT/RPageStorage.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkBuf
\ingroup NTuple
\brief Abstract sink that reorders cluster page writes
*/
// clang-format on
class RPageSinkBuf : public RPageSink {
private:
   std::unique_ptr<RPageSink> fInner;

protected:
   void CreateImpl(const RNTupleModel &model) override;
   RClusterDescriptor::RLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) override;
   RClusterDescriptor::RLocator CommitClusterImpl(NTupleSize_t nEntries) override;
   void CommitDatasetImpl() override;
   // Forward calls to inner descriptor builder
   RNTupleDescriptorBuilder& GetDescriptorBuilder() override { return fInner->fDescriptorBuilder; }

public:
   explicit RPageSinkBuf(std::unique_ptr<RPageSink> inner);
   RPageSinkBuf(const RPageSink&) = delete;
   RPageSinkBuf& operator=(const RPageSinkBuf&) = delete;
   RPageSinkBuf(RPageSinkBuf&&) = default;
   RPageSinkBuf& operator=(RPageSinkBuf&&) = default;
   virtual ~RPageSinkBuf() = default;

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) override;
   void ReleasePage(RPage &page) override;

   RNTupleMetrics &GetMetrics() override { return fInner->GetMetrics(); }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
