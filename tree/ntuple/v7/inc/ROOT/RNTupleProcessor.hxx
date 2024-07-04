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
namespace Internal {

/// Helper type representing the name and storage location of an RNTuple.
struct RNTupleSourceSpec {
   std::string fName;
   std::string fLocation;

   RNTupleSourceSpec() = default;
   RNTupleSourceSpec(std::string_view n, std::string_view s) : fName(n), fLocation(s) {}
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleProcessor
\ingroup NTuple
\brief Interface for iterating over entries of RNTuples and vertically concatenated RNTuples (chains).

Example usage (see ntpl012_processor.C for a full example):

~~~{.cpp}
#include <ROOT/RNTupleProcessor.hxx>
using ROOT::Experimental::RNTupleProcessor;
using ROOT::Experimental::RNTupleSourceSpec;

std::vector<RNTupleSourceSpec> ntuples = {{"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}};
RNTupleProcessor processor(ntuples);
auto ptrPt = processor.GetEntry().GetPtr<float>("pt");

for (const auto &entry : processor) {
   std::cout << "pt = " << *ptrPt << std::endl;
}
~~~

An RNTupleProcessor is created by providing one or more RNTupleSourceSpecs, each of which contains the name and storage
location of a single RNTuple. The RNTuples are processed in the order in which they were provided.

The RNTupleProcessor constructor also (optionally) accepts an RNTupleModel, which determines which fields should be
read. If no model is provided, a default model based on the descriptor of the first specified RNTuple will be used.
If a field that was present in the first RNTuple is not found in a subsequent one, an error will be thrown.

The object returned by the RNTupleProcessor iterator is a view on the current state of the processor, and provides
access to the global entry index (i.e., the entry index taking into account all processed ntuples), local entry index
(i.e. the entry index for only the currently processed ntuple), the index of the ntuple currently being processed (with
respect to the order of provided RNTupleSpecs) and the actual REntry containing the values for the current entry.
*/
// clang-format on
class RNTupleProcessor {
private:
   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleProcessor::RFieldContext
   \ingroup NTuple
   \brief Manager for a field as part of the RNTupleProcessor.

   An RFieldContext contains two fields: a proto-field which is not connected to any page source but serves as the
   blueprint for this particular field, and a concrete field that is connected to the page source currently connected
   to the RNTupleProcessor for reading. When a new page source is connected, the current concrete field gets reset. A
   new concrete field that is connected to this new page source is subsequently created from the proto-field.
   */
   // clang-format on
   class RFieldContext {
      friend class RNTupleProcessor;

   private:
      std::unique_ptr<RFieldBase> fProtoField;
      std::unique_ptr<RFieldBase> fConcreteField;
      REntry::RFieldToken fToken;

   public:
      RFieldContext(std::unique_ptr<RFieldBase> protoField, REntry::RFieldToken token)
         : fProtoField(std::move(protoField)), fToken(token)
      {
      }

      const RFieldBase &GetProtoField() const { return *fProtoField; }
      /// We need to disconnect the concrete fields before swapping the page sources
      void ResetConcreteField() { fConcreteField.reset(); }
      void SetConcreteField() { fConcreteField = fProtoField->Clone(fProtoField->GetFieldName()); }
   };

   std::vector<RNTupleSourceSpec> fNTuples;
   std::unique_ptr<REntry> fEntry;
   std::unique_ptr<Internal::RPageSource> fPageSource;
   std::vector<RFieldContext> fFieldContexts;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect an RNTuple for processing.
   ///
   /// \param[in] ntuple The RNTupleSourceSpec describing the RNTuple to connect.
   ///
   /// \return The number of entries in the newly-connected RNTuple.
   ///
   /// Creates and attaches new page source for the specified RNTuple, and connects the fields that are known by
   /// the processor to it.
   NTupleSize_t ConnectNTuple(const RNTupleSourceSpec &ntuple);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Creates and connects concrete fields to the current page source, based on the proto-fields.
   void ConnectFields();

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Returns a reference to the entry used by the processor.
   ///
   /// \return A reference to the entry used by the processor.
   ///
   const REntry &GetEntry() const { return *fEntry; }

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleProcessor::RIterator
   \ingroup NTuple
   \brief Iterator over the entries of an RNTuple, or vertical concatenation thereof.
   */
   // clang-format on
   class RIterator {
   public:
      // clang-format off
      /**
      \class ROOT::Experimental::RNTupleProcessor::RIterator::RState
      \ingroup NTuple
      \brief View on the RNTupleProcessor iterator state.
      */
      // clang-format on
      class RState {
         friend class RIterator;

      private:
         const REntry &fEntry;
         NTupleSize_t fGlobalEntryIndex;
         NTupleSize_t fLocalEntryIndex;
         /// Index of the currently open RNTuple in the chain of ntuples
         std::size_t fNTupleIndex;

      public:
         RState(const REntry &entry, NTupleSize_t globalEntryIndex, NTupleSize_t localEntryIndex,
                std::size_t ntupleIndex)
            : fEntry(entry),
              fGlobalEntryIndex(globalEntryIndex),
              fLocalEntryIndex(localEntryIndex),
              fNTupleIndex(ntupleIndex)
         {
         }

         const REntry *operator->() const { return &fEntry; }
         const REntry &GetEntry() const { return fEntry; }
         NTupleSize_t GetGlobalEntryIndex() const { return fGlobalEntryIndex; }
         NTupleSize_t GetLocalEntryIndex() const { return fLocalEntryIndex; }
         std::size_t GetNTupleIndex() const { return fNTupleIndex; }
      };

   private:
      RNTupleProcessor &fProcessor;
      RState fState;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = RState;
      using difference_type = std::ptrdiff_t;
      using pointer = RState *;
      using reference = const RState &;

      RIterator(RNTupleProcessor &processor, std::size_t ntupleIndex, NTupleSize_t globalEntryIndex)
         : fProcessor(processor), fState(processor.GetEntry(), globalEntryIndex, 0, ntupleIndex)
      {
      }

      //////////////////////////////////////////////////////////////////////////
      /// \brief Increments the entry index.
      ///
      /// Checks if the end of the currently connected RNTuple is reached. If this is the case, either the next RNTuple
      /// is connected or the iterator has reached the end.
      void Advance()
      {
         ++fState.fGlobalEntryIndex;

         if (++fState.fLocalEntryIndex >= fProcessor.fPageSource->GetNEntries()) {
            do {
               if (++fState.fNTupleIndex >= fProcessor.fNTuples.size()) {
                  fState.fGlobalEntryIndex = kInvalidNTupleIndex;
                  return;
               }
               // Skip over empty ntuples we might encounter.
            } while (fProcessor.ConnectNTuple(fProcessor.fNTuples.at(fState.fNTupleIndex)) == 0);

            fState.fLocalEntryIndex = 0;
         }
         fProcessor.fEntry->Read(fState.fLocalEntryIndex);
      }

      iterator operator++()
      {
         Advance();
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         Advance();
         return obj;
      }

      reference operator*()
      {
         fProcessor.fEntry->Read(fState.fLocalEntryIndex);
         return fState;
      }

      bool operator!=(const iterator &rh) const { return fState.fGlobalEntryIndex != rh.fState.fGlobalEntryIndex; }
      bool operator==(const iterator &rh) const { return fState.fGlobalEntryIndex == rh.fState.fGlobalEntryIndex; }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the descriptor of the first RNTuple specified in `ntuples`.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleProcessor(const std::vector<RNTupleSourceSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);

   RIterator begin() { return RIterator(*this, 0, 0); }
   RIterator end() { return RIterator(*this, fNTuples.size(), kInvalidNTupleIndex); }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleProcessor
