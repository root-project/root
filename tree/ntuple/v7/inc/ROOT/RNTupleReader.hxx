/// \file ROOT/RNTupleReader.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleReader
#define ROOT7_RNTupleReader

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RSpan.hxx>

#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

namespace ROOT {
class RNTuple;

namespace Experimental {

/// Listing of the different options that can be printed by RNTupleReader::GetInfo()
enum class ENTupleInfo {
   kSummary,        // The ntuple name, description, number of entries
   kStorageDetails, // size on storage, page sizes, compression factor, etc.
   kMetrics,        // internals performance counters, requires that EnableMetrics() was called
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleReader
\ingroup NTuple
\brief An RNTuple that is used to read data from storage

An input ntuple provides data from storage as C++ objects. The ntuple model can be created from the data on storage
or it can be imposed by the user. The latter case allows users to read into a specialized ntuple model that covers
only a subset of the fields in the ntuple. The ntuple model is used when reading complete entries.
Individual fields can be read as well by instantiating a tree view.

~~~ {.cpp}
#include <ROOT/RNTupleReader.hxx>
using ROOT::Experimental::RNTupleReader;

#include <iostream>

auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
std::cout << "myNTuple has " << ntuple->GetNEntries() << " entries\n";
~~~
*/
// clang-format on
class RNTupleReader {
private:
   /// Set as the page source's scheduler for parallel page decompression if IMT is on
   /// Needs to be destructed after the pages source is destructed (an thus be declared before)
   std::unique_ptr<Internal::RPageStorage::RTaskScheduler> fUnzipTasks;

   std::unique_ptr<Internal::RPageSource> fSource;
   /// Needs to be destructed before fSource
   std::unique_ptr<RNTupleModel> fModel;
   /// We use a dedicated on-demand reader for Show() and Scan(). Printing data uses all the fields
   /// from the full model even if the analysis code uses only a subset of fields. The display reader
   /// is a clone of the original reader.
   std::unique_ptr<RNTupleReader> fDisplayReader;
   /// The ntuple descriptor in the page source is protected by a read-write lock. We don't expose that to the
   /// users of RNTupleReader::GetDescriptor().  Instead, if descriptor information is needed, we clone the
   /// descriptor.  Using the descriptor's generation number, we know if the cached descriptor is stale.
   /// Retrieving descriptor data from an RNTupleReader is supposed to be for testing and information purposes,
   /// not on a hot code path.
   std::optional<RNTupleDescriptor> fCachedDescriptor;
   Detail::RNTupleMetrics fMetrics;
   /// If not nullopt, these will used when creating the model
   std::optional<RNTupleDescriptor::RCreateModelOptions> fCreateModelOptions;

   RNTupleReader(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSource> source,
                 const ROOT::RNTupleReadOptions &options);
   /// The model is generated from the ntuple metadata on storage.
   explicit RNTupleReader(std::unique_ptr<Internal::RPageSource> source, const ROOT::RNTupleReadOptions &options);

   void ConnectModel(RNTupleModel &model);
   RNTupleReader *GetDisplayReader();
   void InitPageSource(bool enableMetrics);

   ROOT::DescriptorId_t RetrieveFieldId(std::string_view fieldName) const;

public:
   // Browse through the entries
   class RIterator {
   private:
      ROOT::NTupleSize_t fIndex = ROOT::kInvalidNTupleIndex;

   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = ROOT::NTupleSize_t;
      using difference_type = ROOT::NTupleSize_t;
      using pointer = ROOT::NTupleSize_t *;
      using reference = ROOT::NTupleSize_t &;

      RIterator() = default;
      explicit RIterator(ROOT::NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         fIndex++;
         return r;
      }
      iterator &operator++() /* prefix */
      {
         ++fIndex;
         return *this;
      }
      reference operator*() { return fIndex; }
      pointer operator->() { return &fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
   };

   /// Open an RNTuple for reading.
   ///
   /// Throws an RException if there is no RNTuple with the given name.
   ///
   /// **Example: open an RNTuple and print the number of entries**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// std::cout << "myNTuple has " << ntuple->GetNEntries() << " entries\n";
   /// ~~~
   static std::unique_ptr<RNTupleReader> Open(std::string_view ntupleName, std::string_view storage,
                                              const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());
   static std::unique_ptr<RNTupleReader>
   Open(const RNTuple &ntuple, const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());

   /// The caller imposes a model, which must be compatible with the model found in the data on storage.
   static std::unique_ptr<RNTupleReader> Open(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                              std::string_view storage,
                                              const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());
   static std::unique_ptr<RNTupleReader> Open(std::unique_ptr<RNTupleModel> model, const RNTuple &ntuple,
                                              const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());

   /// The caller imposes the way the model is reconstructed
   static std::unique_ptr<RNTupleReader> Open(const RNTupleDescriptor::RCreateModelOptions &createModelOpts,
                                              std::string_view ntupleName, std::string_view storage,
                                              const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());
   static std::unique_ptr<RNTupleReader> Open(const RNTupleDescriptor::RCreateModelOptions &createModelOpts,
                                              const RNTuple &ntuple,
                                              const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());
   std::unique_ptr<RNTupleReader> Clone()
   {
      auto options = ROOT::RNTupleReadOptions{};
      options.SetEnableMetrics(fMetrics.IsEnabled());
      return std::unique_ptr<RNTupleReader>(new RNTupleReader(fSource->Clone(), options));
   }
   ~RNTupleReader();

   ROOT::NTupleSize_t GetNEntries() const { return fSource->GetNEntries(); }
   const RNTupleModel &GetModel();
   std::unique_ptr<REntry> CreateEntry();

   /// Returns a cached copy of the page source descriptor. The returned pointer remains valid until the next call
   /// to LoadEntry or to any of the views returned from the reader.
   const RNTupleDescriptor &GetDescriptor();

   /// Prints a detailed summary of the ntuple, including a list of fields.
   ///
   /// **Example: print summary information to stdout**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// ntuple->PrintInfo();
   /// // or, equivalently:
   /// ntuple->PrintInfo(ENTupleInfo::kSummary, std::cout);
   /// ~~~
   /// **Example: print detailed column storage data to stderr**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// ntuple->PrintInfo(ENTupleInfo::kStorageDetails, std::cerr);
   /// ~~~
   ///
   /// For use of ENTupleInfo::kMetrics, see #EnableMetrics.
   void PrintInfo(const ENTupleInfo what = ENTupleInfo::kSummary, std::ostream &output = std::cout) const;

   /// Shows the values of the i-th entry/row, starting with 0 for the first entry. By default,
   /// prints the output in JSON format.
   /// Uses the visitor pattern to traverse through each field of the given entry.
   void Show(ROOT::NTupleSize_t index, std::ostream &output = std::cout);

   /// Analogous to Fill(), fills the default entry of the model. Returns false at the end of the ntuple.
   /// On I/O errors, raises an exception.
   void LoadEntry(ROOT::NTupleSize_t index)
   {
      // TODO(jblomer): can be templated depending on the factory method / constructor
      if (R__unlikely(!fModel)) {
         fModel = fSource->GetSharedDescriptorGuard()->CreateModel(
            fCreateModelOptions.value_or(RNTupleDescriptor::RCreateModelOptions{}));
         ConnectModel(*fModel);
      }
      LoadEntry(index, fModel->GetDefaultEntry());
   }
   /// Fills a user provided entry after checking that the entry has been instantiated from the ntuple model
   void LoadEntry(ROOT::NTupleSize_t index, REntry &entry)
   {
      if (R__unlikely(entry.GetModelId() != fModel->GetModelId()))
         throw RException(R__FAIL("mismatch between entry and model"));

      entry.Read(index);
   }

   /// Returns an iterator over the entry indices of the RNTuple.
   ///
   /// **Example: iterate over all entries and print each entry in JSON format**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::ENTupleShowFormat;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    ntuple->Show(i);
   /// }
   /// ~~~
   ROOT::RNTupleGlobalRange GetEntryRange() { return ROOT::RNTupleGlobalRange(0, GetNEntries()); }

   /// Provides access to an individual field that can contain either a scalar value or a collection, e.g.
   /// GetView<double>("particles.pt") or GetView<std::vector<double>>("particle").  It can as well be the index
   /// field of a collection itself, like GetView<ROOT::NTupleSize_t>("particle").
   ///
   /// Raises an exception if there is no field with the given name.
   ///
   /// **Example: iterate over a field named "pt" of type `float`**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// auto pt = ntuple->GetView<float>("pt");
   ///
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    std::cout << i << ": " << pt(i) << "\n";
   /// }
   /// ~~~
   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName)
   {
      return GetView<T>(RetrieveFieldId(fieldName));
   }

   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName, std::shared_ptr<T> objPtr)
   {
      return GetView<T>(RetrieveFieldId(fieldName), objPtr);
   }

   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName, T *rawPtr)
   {
      return GetView<T>(RetrieveFieldId(fieldName), rawPtr);
   }

   template <typename T>
   RNTupleView<T> GetView(ROOT::DescriptorId_t fieldId)
   {
      auto field = RNTupleView<T>::CreateField(fieldId, *fSource);
      auto range = Internal::GetFieldRange(*field, *fSource);
      return RNTupleView<T>(std::move(field), range);
   }

   template <typename T>
   RNTupleView<T> GetView(ROOT::DescriptorId_t fieldId, std::shared_ptr<T> objPtr)
   {
      auto field = RNTupleView<T>::CreateField(fieldId, *fSource);
      auto range = Internal::GetFieldRange(*field, *fSource);
      return RNTupleView<T>(std::move(field), range, objPtr);
   }

   template <typename T>
   RNTupleView<T> GetView(ROOT::DescriptorId_t fieldId, T *rawPtr)
   {
      auto field = RNTupleView<T>::CreateField(fieldId, *fSource);
      auto range = Internal::GetFieldRange(*field, *fSource);
      return RNTupleView<T>(std::move(field), range, rawPtr);
   }

   template <typename T>
   RNTupleDirectAccessView<T> GetDirectAccessView(std::string_view fieldName)
   {
      return GetDirectAccessView<T>(RetrieveFieldId(fieldName));
   }

   template <typename T>
   RNTupleDirectAccessView<T> GetDirectAccessView(ROOT::DescriptorId_t fieldId)
   {
      auto field = RNTupleDirectAccessView<T>::CreateField(fieldId, *fSource);
      auto range = Internal::GetFieldRange(field, *fSource);
      return RNTupleDirectAccessView<T>(std::move(field), range);
   }

   /// Raises an exception if:
   /// * there is no field with the given name or,
   /// * the field is not a collection
   RNTupleCollectionView GetCollectionView(std::string_view fieldName)
   {
      auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName);
      if (fieldId == ROOT::kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                                  fSource->GetSharedDescriptorGuard()->GetName() + "'"));
      }
      return GetCollectionView(fieldId);
   }

   RNTupleCollectionView GetCollectionView(ROOT::DescriptorId_t fieldId)
   {
      return RNTupleCollectionView::Create(fieldId, fSource.get());
   }

   RIterator begin() { return RIterator(0); }
   RIterator end() { return RIterator(GetNEntries()); }

   /// Enable performance measurements (decompression time, bytes read from storage, etc.)
   ///
   /// **Example: inspect the reader metrics after loading every entry**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTupleReader.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// // metrics must be turned on beforehand
   /// ntuple->EnableMetrics();
   ///
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    ntuple->LoadEntry(i);
   /// }
   /// ntuple->PrintInfo(ENTupleInfo::kMetrics);
   /// ~~~
   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
}; // class RNTupleReader

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleReader
