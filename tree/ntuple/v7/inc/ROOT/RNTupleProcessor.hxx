/// \file ROOT/RNTupleProcessor.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-03-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleProcessor
#define ROOT7_RNTupleProcessor

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RSpan.hxx>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {
namespace Experimental {

class RNTupleProcessor {
public:
   struct RProcessorSpec {
      std::string fNTupleName;
      std::string fStorage;

      RProcessorSpec() = default;
      RProcessorSpec(std::string_view n, std::string_view s) : fNTupleName(n), fStorage(s) {}
   };

   struct RProcessorField {
      std::unique_ptr<ROOT::Experimental::RFieldBase> fProtoField;
      std::unique_ptr<ROOT::Experimental::RFieldBase> fConcreteField;
      REntry::RFieldToken fToken;
      std::shared_ptr<void> fValuePtr;

      RProcessorField(std::unique_ptr<ROOT::Experimental::RFieldBase> protoField, REntry::RFieldToken token)
         : fProtoField(std::move(protoField)), fToken(token)
      {
      }

      void ResetConcreteField() { fConcreteField = fProtoField->Clone(fProtoField->GetFieldName()); }
   };

private:
   std::vector<std::unique_ptr<Internal::RPageSource>> fPageSources;

   std::unique_ptr<REntry> fEntry; ///< The entry is based on the first page source

   std::vector<RProcessorField> fProcessorFields;

   void SetProcessorFields(Internal::RPageSource &pageSource, REntry &entry, DescriptorId_t fieldId);
   void ConnectFields(Internal::RPageSource &pageSource);

public:
   class RIterator {
   private:
      RNTupleProcessor &fProcessor;
      std::size_t fPageSourceIdx;
      NTupleSize_t fGlobalEntryIndex;
      NTupleSize_t fLocalEntryIndex;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = REntry;
      using difference_type = std::ptrdiff_t;
      using pointer = REntry *;
      using reference = const REntry &;

      RIterator(RNTupleProcessor &processor, std::size_t pageSourceIdx, NTupleSize_t globalEntryIndex)
         : fProcessor(processor),
           fPageSourceIdx(pageSourceIdx),
           fGlobalEntryIndex(globalEntryIndex),
           fLocalEntryIndex(0)
      {
      }

      iterator operator++()
      {
         ++fGlobalEntryIndex;

         if (++fLocalEntryIndex >= fProcessor.fPageSources.at(fPageSourceIdx)->GetNEntries()) {
            if (++fPageSourceIdx >= fProcessor.fPageSources.size()) {
               fGlobalEntryIndex = kInvalidNTupleIndex;
               return *this;
            }

            auto &pageSource = fProcessor.fPageSources.at(fPageSourceIdx);
            pageSource->Attach();
            fProcessor.ConnectFields(*pageSource);
            fLocalEntryIndex = 0;
         }
         return *this;
      }

      reference operator*()
      {
         fProcessor.fEntry->Read(fLocalEntryIndex);
         return *fProcessor.fEntry;
      }
      bool operator!=(const iterator &rh) const { return fGlobalEntryIndex != rh.fGlobalEntryIndex; }
      bool operator==(const iterator &rh) const { return fGlobalEntryIndex == rh.fGlobalEntryIndex; }
   };

   RNTupleProcessor(const RNTupleProcessor &other) = delete;
   RNTupleProcessor &operator=(const RNTupleProcessor &other) = delete;
   RNTupleProcessor(RNTupleProcessor &&other) = delete;
   RNTupleProcessor &operator=(RNTupleProcessor &&other) = delete;
   ~RNTupleProcessor() = default;

   RNTupleProcessor(std::span<RProcessorSpec> ntuples);

   RIterator begin() { return RIterator(*this, 0, 0); }
   RIterator end() { return RIterator(*this, fPageSources.size(), kInvalidNTupleIndex); }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleProcessor
