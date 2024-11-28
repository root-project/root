/// \file
/// \ingroup tutorial_ntuple
///
/// Example of a streaming vector: a special purpose container that reads large vectors piece-wise.
///
/// \macro_code
///
/// \date November 2024
/// \author Peter van Gemmeren, the ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TRandom3.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

using namespace ROOT::Experimental;

constexpr char const *kFileName = "ntpl015_streaming_vector.root";
constexpr char const *kNTupleName = "ntpl";
constexpr char const *kFieldName = "LargeVector";
constexpr unsigned int kNEvents = 10;
constexpr unsigned int kVectorSize = 1000000;

void CreateRNTuple()
{
   auto model = RNTupleModel::Create();
   auto ptrLargeVector = model->MakeField<std::vector<std::uint32_t>>(kFieldName);
   auto writer = RNTupleWriter::Recreate(std::move(model), kNTupleName, kFileName);

   auto prng = std::make_unique<TRandom3>();
   prng->SetSeed();

   for (NTupleSize_t i = 0; i < kNEvents; i++) {
      ptrLargeVector->clear();
      for (std::size_t j = 0; j < kVectorSize; j++)
         ptrLargeVector->emplace_back(prng->Integer(-1));
      writer->Fill();
   }
   std::cout << "RNTuple written" << std::endl;
}

/*
 * ==================================================================================================
 */

void ReadRNTupleSimple()
{
   auto reader = RNTupleReader::Open(kNTupleName, kFileName);

   const auto nEntries = reader->GetNEntries();
   std::cout << "Simple reading, found " << nEntries << " entries" << std::endl;

   auto ptrLargeVector = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::uint32_t>>(kFieldName);
   for (NTupleSize_t i = 0; i < nEntries; i++) {
      reader->LoadEntry(i);

      const auto vectorSize = ptrLargeVector->size();
      uint64_t sum = 0;
      for (auto val : *ptrLargeVector)
         sum += val;

      std::cout << "Size and sum of vector: " << vectorSize << " " << sum << std::endl;
   }
   std::cout << "RNTuple simple read" << std::endl;
}

/*
 * ==================================================================================================
 */

template <class T>
class RStreamingVector {
   RNTupleCollectionView fVectorView;
   RNTupleView<T> fItemView;
   RNTupleClusterRange fRange;
   NTupleSize_t fEntry{0};
   NTupleSize_t fSize{0};

public:
   class iterator {
      RNTupleClusterRange::RIterator fIndex;
      RNTupleView<T> &fView;

   public:
      iterator(RNTupleClusterRange::RIterator index, RNTupleView<T> &view) : fIndex(index), fView(view) {}
      ~iterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         ++(*this);
         return r;
      }
      iterator &operator++() /* prefix */
      {
         ++fIndex;
         return *this;
      }
      const T &operator*() { return fView.operator()(*fIndex); }
      const T *operator->() { return &fView.operator()(*fIndex); }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
   };

   RStreamingVector(RNTupleCollectionView &&vectorView)
      : fVectorView(std::move(vectorView)), fItemView(fVectorView.GetView<T>("_0"))
   {
   }
   ~RStreamingVector() = default;

   NTupleSize_t size() const { return fSize; }

   iterator begin() { return iterator(fRange.begin(), fItemView); }
   iterator end() { return iterator(fRange.end(), fItemView); }

   void LoadEntry(NTupleSize_t entry)
   {
      fEntry = entry;
      fRange = fVectorView.GetCollectionRange(fEntry);
      fSize = fVectorView.operator()(fEntry);
   }
};

void ReadRNTupleStreamingVector()
{
   auto reader = RNTupleReader::Open(kNTupleName, kFileName);

   const auto nEntries = reader->GetNEntries();
   std::cout << "Streamed reading, found " << nEntries << " entries" << std::endl;

   RStreamingVector<std::uint32_t> streamingVector(reader->GetCollectionView(kFieldName));
   for (NTupleSize_t i = 0; i < nEntries; i++) {
      streamingVector.LoadEntry(i);

      const auto vectorSize = streamingVector.size();
      uint64_t sum = 0;
      for (auto val : streamingVector)
         sum += val;

      std::cout << "Size and sum of vector: " << vectorSize << " " << sum << std::endl;
   }
   std::cout << "RNTuple streaming read" << std::endl;
}

void ntpl015_streaming_vector()
{
   CreateRNTuple();
   ReadRNTupleSimple();
   ReadRNTupleStreamingVector();
}
