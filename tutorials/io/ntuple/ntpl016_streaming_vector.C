/// \file
/// \ingroup tutorial_ntuple
///
/// Example of a streaming vector: a special purpose container that reads large vectors piece-wise.
///
/// \macro_code
///
/// \date November 2024
/// \author Peter van Gemmeren, the ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TRandom3.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

constexpr char const *kFileName = "ntpl016_streaming_vector.root";
constexpr char const *kNTupleName = "ntpl";
constexpr char const *kFieldName = "LargeVector";
constexpr unsigned int kNEvents = 10;
constexpr unsigned int kNElementsPerVector = 1000000;

// Create an RNTuple with a single vector field. Every entry contains a large vector of random integers.
// The vector should be seen as too large to be held entirely in memory during reading.
void CreateRNTuple()
{
   auto model = ROOT::RNTupleModel::Create();
   auto ptrLargeVector = model->MakeField<std::vector<std::uint32_t>>(kFieldName);
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), kNTupleName, kFileName);

   auto prng = std::make_unique<TRandom3>();
   prng->SetSeed();

   for (ROOT::NTupleSize_t i = 0; i < kNEvents; i++) {
      ptrLargeVector->clear();
      for (std::size_t j = 0; j < kNElementsPerVector; j++)
         ptrLargeVector->emplace_back(prng->Integer(-1));
      writer->Fill();
   }
   std::cout << "RNTuple written" << std::endl;
}

/*
 * ==================================================================================================
 */

// For comparison, the canonical read function that reads the entire vector for every entry.
void ReadRNTupleSimple()
{
   auto reader = ROOT::RNTupleReader::Open(kNTupleName, kFileName);

   const auto nEntries = reader->GetNEntries();
   std::cout << "Simple reading, found " << nEntries << " entries" << std::endl;

   auto ptrLargeVector = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::uint32_t>>(kFieldName);
   for (ROOT::NTupleSize_t i = 0; i < nEntries; i++) {
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

// The StreamingVectorView class allows iteration over an RNTuple on-disk vector of element type T.
// Unlike an std::vector, this class does not provide random-access but only allows to iterate the data elements
// from beginning to end.
// Internally, it uses an RNTupleCollection view and an item view to load chunks of the vector elements into memory,
// so that never the entire vector needs to stay in memory.
// Note that we don't need to implement loading chunks of data explicitly. Simply by asking for a single vector element
// at every iteration step, the RNTuple views will take care of keeping only the currently required data pages
// in memory. This results in the minimal possible memory footprint of RNTuple.
// Note that for effective streaming, the cluster cache read option needs to be turned off. This may change in the
// future with more fine-grained control of the data preloading.
template <class T>
class StreamingVectorView {
   // For a certain entry, the collection view provides the information about the size of the collection and
   // the index range of the item view, which is required to read the values of the collection at hand.
   ROOT::RNTupleCollectionView fVectorView;
   // The "data view" provides access to the vector elements
   ROOT::RNTupleView<T> fItemView;
   // Given an entry number, the start end end index in the item view to read the corresponding vector elements
   ROOT::RNTupleLocalRange fRange{ROOT::kInvalidDescriptorId, ROOT::kInvalidNTupleIndex, ROOT::kInvalidNTupleIndex};
   // The index of the entry from which the vector should be read
   ROOT::NTupleSize_t fEntry{0};
   // The size of the collection in fEntry
   ROOT::NTupleSize_t fSize{0};

public:
   // A lightweight iterator used in StreamingVectorView::begin() and StreamingVectorView::end().
   // Used to iterate over the elements of an RNTuple on-disk vector for a certain entry.
   // Dereferencing the iterator returns the corresponding value of the item view.
   class Iterator {
      ROOT::RNTupleLocalRange::RIterator fRangeItr;
      ROOT::RNTupleView<T> &fView;

   public:
      using iterator = Iterator;
      using iterator_category = std::input_iterator_tag;
      using value_type = T;
      using pointer = const T *;
      using reference = const T &;

      Iterator(ROOT::RNTupleLocalRange::RIterator rangeItr, ROOT::RNTupleView<T> &view)
         : fRangeItr(rangeItr), fView(view)
      {
      }

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         ++(*this);
         return r;
      }
      iterator &operator++() /* prefix */
      {
         ++fRangeItr;
         return *this;
      }
      reference operator*() { return fView.operator()(*fRangeItr); }
      pointer operator->() { return &fView.operator()(*fRangeItr); }
      bool operator==(const iterator &rh) const { return fRangeItr == rh.fRangeItr; }
      bool operator!=(const iterator &rh) const { return fRangeItr != rh.fRangeItr; }
   };

   explicit StreamingVectorView(ROOT::RNTupleCollectionView vectorView)
      : fVectorView(std::move(vectorView)), fItemView(fVectorView.GetView<T>("_0"))
   {
   }

   ROOT::NTupleSize_t size() const { return fSize; }

   // The begin() and end() methods enable range-based for loops like `for (auto val : streamingVector)`
   Iterator begin() { return Iterator(fRange.begin(), fItemView); }
   Iterator end() { return Iterator(fRange.end(), fItemView); }

   void LoadEntry(ROOT::NTupleSize_t entry)
   {
      fEntry = entry;
      fRange = fVectorView.GetCollectionRange(fEntry);
      fSize = fVectorView.operator()(fEntry);
   }
};

// For the streaming vector read, we use a custom class `StreamingVectorView` that implements the piece-wise
// loading of the data during iteration of elements of the on-disk vector. The class has been built such that
// the event loop is almost identical to the simple reading case above.
void ReadRNTupleStreamingVector()
{
   ROOT::RNTupleReadOptions options;
   // Don't preload data; we want to populate data into memory only as needed
   options.SetClusterCache(ROOT::RNTupleReadOptions::EClusterCache::kOff);
   auto reader = ROOT::RNTupleReader::Open(kNTupleName, kFileName, options);

   const auto nEntries = reader->GetNEntries();
   std::cout << "Streamed reading, found " << nEntries << " entries" << std::endl;

   StreamingVectorView<std::uint32_t> streamingVector(reader->GetCollectionView(kFieldName));

   for (ROOT::NTupleSize_t i = 0; i < nEntries; i++) {
      // Instead of `reader->LoadEntry()`, we tell the streaming vector which entry we want to read.
      streamingVector.LoadEntry(i);

      // We can ask for the size of the vector without loading the data
      const auto vectorSize = streamingVector.size();

      // The iteration works exactly as in the simple case
      uint64_t sum = 0;
      for (auto val : streamingVector)
         sum += val;

      std::cout << "Size and sum of vector: " << vectorSize << " " << sum << std::endl;
   }
   std::cout << "RNTuple streaming read" << std::endl;
}

void ntpl016_streaming_vector()
{
   CreateRNTuple();
   ReadRNTupleSimple();
   ReadRNTupleStreamingVector();
}
