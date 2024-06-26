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
auto ptrPt = processor.GetPtr<float>("pt");

for (const auto &entry : processor) {
   std::cout << "pt = " << *ptrPt << std::endl;
}
~~~

An RNTupleProcessor is created by providing one or more RNTupleSourceSpecs, each of which contains the name and storage
location of a single RNTuple. The RNTuples are subsequently processed in the order in which they were provided.

The default model of the RNTuple that is provided first will be used to construct the fields in the entry that will be
filled in each iteration. This entry is owned by the RNTupleProcessor.
If subsequently processed RNTuples contain additional fields, they will be ignored.
Conversely, if a field that was present in the first RNTuple is not found in a subsequent one, an error will be thrown.

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
   to the RNTupleProcessor object for reading. When a new page source is connected, the concrete field gets reset by
   cloning the proto-field and connecting it to the new page source.

   Apart from the fields themselves, the RFieldContext object also manages the pointer to the value for this particular
   field that's read by the processor.
   */
   // clang-format on
   class RFieldContext {
      friend class RNTupleProcessor;

   private:
      std::unique_ptr<RFieldBase> fProtoField;
      std::unique_ptr<RFieldBase> fConcreteField;
      REntry::RFieldToken fToken;
      std::shared_ptr<void> fValuePtr;

   public:
      RFieldContext(std::unique_ptr<RFieldBase> protoField, REntry::RFieldToken token)
         : fProtoField(std::move(protoField)), fToken(token)
      {
      }

      const RFieldBase &GetProtoField() const { return *fProtoField; }
      /// We need to disconnect the concrete fields before swapping the page sources
      void ResetConcreteField() { fConcreteField = nullptr; }
      RFieldBase &CreateConcreteField()
      {
         fConcreteField = fProtoField->Clone(fProtoField->GetFieldName());
         return *fConcreteField;
      }
      RFieldBase &GetConcreteField() const { return *fConcreteField; }
      const REntry::RFieldToken &GetToken() const { return fToken; }
   };

   std::vector<RNTupleSourceSpec> fNTuples;
   std::unique_ptr<REntry> fEntry;
   std::unique_ptr<Internal::RPageSource> fPageSource;
   std::unordered_map<std::string, RFieldContext> fFieldContexts;

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

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to be read by the processor.
   ///
   /// \param[in] fieldName The name of the field to activate.
   void ActivateField(std::string_view fieldName);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field has been activated in the processor.
   ///
   /// \param[in] fieldName The name of the field to check.
   ///
   /// \return Whether the field with the given name is activated.
   bool HasField(std::string_view fieldName) { return fFieldContexts.count(std::string(fieldName)) > 0; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the values for the provided entry index.
   ///
   /// \param[in] idx The entry index.
   void LoadEntry(NTupleSize_t idx) { fEntry->Read(idx); }

public:
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
         friend class RNTupleProcessor;

      private:
         RNTupleProcessor &fProcessor;
         NTupleSize_t fGlobalEntryIndex;
         NTupleSize_t fLocalEntryIndex;
         /// Index of the currently open RNTuple in the chain of ntuples
         std::size_t fNTupleIndex;

         void UpdateEntry() { fProcessor.LoadEntry(fLocalEntryIndex); }

      public:
         RState(RNTupleProcessor &processor, NTupleSize_t globalEntryIndex, NTupleSize_t localEntryIndex,
                std::size_t ntupleIndex)
            : fProcessor(processor),
              fGlobalEntryIndex(globalEntryIndex),
              fLocalEntryIndex(localEntryIndex),
              fNTupleIndex(ntupleIndex)
         {
         }

         NTupleSize_t GetGlobalEntryIndex() const { return fGlobalEntryIndex; }
         NTupleSize_t GetLocalEntryIndex() const { return fLocalEntryIndex; }
         std::size_t GetNTupleIndex() const { return fNTupleIndex; }

         template <typename T>
         std::shared_ptr<T> GetPtr(std::string_view fieldName)
         {
            if (!fProcessor.HasField(fieldName)) {
               fProcessor.ActivateField(fieldName);
               UpdateEntry();
            }
            return fProcessor.GetPtr<T>(fieldName);
         }
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
      using reference = RState &;

      RIterator(RNTupleProcessor &processor, std::size_t ntupleIndex, NTupleSize_t globalEntryIndex)
         : fProcessor(processor), fState(processor, globalEntryIndex, 0, ntupleIndex)
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
         fState.UpdateEntry();
         return fState;
      }

      pointer operator->()
      {
         fState.UpdateEntry();
         return &fState;
      }

      bool operator!=(const iterator &rh) const { return fState.fGlobalEntryIndex != rh.fState.fGlobalEntryIndex; }
      bool operator==(const iterator &rh) const { return fState.fGlobalEntryIndex == rh.fState.fGlobalEntryIndex; }
   };

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleProcessor(const std::vector<RNTupleSourceSpec> &ntuples);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the names of the fields currently actively being processed.
   ///
   /// \return A vector with the names of all currently active fields.
   const std::vector<std::string> GetActiveFields() const;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Returns a pointer to the object representing the provided field during processing.
   ///
   /// \param[in] fieldName the name of the field for which to get a pointer.
   ///
   /// \return A pointer to the object for the provided field.
   ///
   /// \warning If this method is called while iterating, the pointer won't hold an entry value until the *next*
   /// iteration step. To ensure the pointer holds the correct entry value during iteration, use
   /// RNTupleProcessor::RIterator::RState::GetPtr instead.
   template <typename T>
   std::shared_ptr<T> GetPtr(std::string_view fieldName)
   {
      if (!HasField(fieldName))
         ActivateField(fieldName);
      return fEntry->GetPtr<T>(fieldName);
   }

   RIterator begin() { return RIterator(*this, 0, 0); }
   RIterator end() { return RIterator(*this, fNTuples.size(), kInvalidNTupleIndex); }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleProcessor
