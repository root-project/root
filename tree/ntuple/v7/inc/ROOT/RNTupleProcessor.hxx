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
auto processor = RNTupleProcessor::CreateChain(ntuples);

for (const auto &entry : processor) {
   std::cout << "pt = " << *entry.GetPtr<float>("pt") << std::endl;
}
~~~

An RNTupleProcessor is created by providing one or more RNTupleSourceSpecs, each of which contains the name and storage
location of a single RNTuple. The RNTuples are processed in the order in which they were provided.

The RNTupleProcessor constructor also (optionally) accepts an RNTupleModel, which determines which fields should be
read. If no model is provided, a default model based on the descriptor of the first specified RNTuple will be used.
If a field that was present in the first RNTuple is not found in a subsequent one, an error will be thrown.

The RNTupleProcessor provides an iterator which gives access to the REntry containing the field data for the current
entry. Additional bookkeeping information can be obtained through the RNTupleProcessor itself.
*/
// clang-format on
class RNTupleProcessor {
protected:
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
      friend class RNTupleChainProcessor;

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

   NTupleSize_t fNEntriesProcessed;  //< Total number of entries processed so far
   std::size_t fCurrentNTupleNumber; //< Index of the currently open RNTuple
   NTupleSize_t fLocalEntryNumber;   //< Entry number within the current ntuple

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect an RNTuple for processing.
   ///
   /// \param[in] ntuple The RNTupleSourceSpec describing the RNTuple to connect.
   ///
   /// \return The number of entries in the newly-connected RNTuple.
   ///
   /// Creates and attaches new page source for the specified RNTuple, and connects the fields that are known by
   /// the processor to it.
   virtual NTupleSize_t ConnectNTuple(const RNTupleSourceSpec &ntuple) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Creates and connects concrete fields to the current page source, based on the proto-fields.
   virtual void ConnectFields() = 0;

   //////////////////////////////////////////////////////////////////////////
   /// \brief Advance the processor to the next available entry.
   ///
   /// \return The new (global) entry number of after advancing, or kInvalidNTupleIndex if the last entry has been
   /// processed.
   ///
   /// Checks if the end of the currently connected RNTuple is reached. If this is the case, either the next RNTuple
   /// is connected or the iterator has reached the end.
   virtual NTupleSize_t Advance() = 0;

   RNTupleProcessor(const std::vector<RNTupleSourceSpec> &ntuples)
      : fNTuples(ntuples), fNEntriesProcessed(0), fCurrentNTupleNumber(0), fLocalEntryNumber(0)
   {
   }

public:
   RNTupleProcessor(const RNTupleProcessor &) = delete;
   RNTupleProcessor(RNTupleProcessor &&) = delete;
   RNTupleProcessor &operator=(const RNTupleProcessor &) = delete;
   RNTupleProcessor &operator=(RNTupleProcessor &&) = delete;
   virtual ~RNTupleProcessor() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries processed so far.
   ///
   /// When only one RNTuple is present in the processor chain, the return value is equal to GetLocalEntryNumber.
   NTupleSize_t GetNEntriesProcessed() const { return fNEntriesProcessed; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the index to the RNTuple currently being processed, according to the sources specified upon creation.
   std::size_t GetCurrentNTupleNumber() const { return fCurrentNTupleNumber; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number local to the RNTuple that is currently being processed.
   ///
   /// When only one RNTuple is present in the processor chain, the return value is equal to GetGlobalEntryNumber.
   NTupleSize_t GetLocalEntryNumber() const { return fLocalEntryNumber; }

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
   private:
      RNTupleProcessor &fProcessor;
      NTupleSize_t fNEntriesProcessed;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = REntry;
      using difference_type = std::ptrdiff_t;
      using pointer = REntry *;
      using reference = const REntry &;

      RIterator(RNTupleProcessor &processor, NTupleSize_t globalEntryNumber)
         : fProcessor(processor), fNEntriesProcessed(globalEntryNumber)
      {
      }

      iterator operator++()
      {
         fNEntriesProcessed = fProcessor.Advance();
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         obj.fNEntriesProcessed = fProcessor.Advance();
         return obj;
      }

      reference operator*()
      {
         fProcessor.fEntry->Read(fProcessor.fLocalEntryNumber);
         return *fProcessor.fEntry;
      }

      friend bool operator!=(const iterator &lh, const iterator &rh)
      {
         return lh.fNEntriesProcessed != rh.fNEntriesProcessed;
      }
      friend bool operator==(const iterator &lh, const iterator &rh)
      {
         return lh.fNEntriesProcessed == rh.fNEntriesProcessed;
      }
   };

   RIterator begin() { return RIterator(*this, 0); }
   RIterator end() { return RIterator(*this, kInvalidNTupleIndex); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new RNTuple processor chain for vertical concatenation of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the ntuples to process.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first ntuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateChain(const std::vector<RNTupleSourceSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleChainProcessor
\ingroup NTuple
\brief Processor specializiation for vertically concatenated RNTuples (chains).
*/
// clang-format on
class RNTupleChainProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   NTupleSize_t ConnectNTuple(const RNTupleSourceSpec &ntuple) final;
   void ConnectFields() final;
   NTupleSize_t Advance() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleChainProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the descriptor of the first RNTuple specified in `ntuples`.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainProcessor(const std::vector<RNTupleSourceSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleProcessor
