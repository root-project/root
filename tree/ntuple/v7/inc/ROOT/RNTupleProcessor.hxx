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
#include <ROOT/RNTupleIndex.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {
namespace Experimental {

/// Used to specify the underlying RNTuples in RNTupleProcessor
struct RNTupleOpenSpec {
   std::string fNTupleName;
   std::string fStorage;
   RNTupleReadOptions fOptions;

   RNTupleOpenSpec(std::string_view n, std::string_view s) : fNTupleName(n), fStorage(s) {}
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
using ROOT::Experimental::RNTupleOpenSpec;

std::vector<RNTupleOpenSpec> ntuples = {{"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}};
auto processor = RNTupleProcessor::CreateChain(ntuples);

for (const auto &entry : processor) {
   std::cout << "pt = " << *entry.GetPtr<float>("pt") << std::endl;
}
~~~

An RNTupleProcessor is created by providing one or more RNTupleOpenSpecs, each of which contains the name and storage
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
      friend class RNTupleSingleProcessor;
      friend class RNTupleChainProcessor;
      friend class RNTupleJoinProcessor;

   private:
      std::unique_ptr<RFieldBase> fProtoField;
      std::unique_ptr<RFieldBase> fConcreteField;
      REntry::RFieldToken fToken;
      // Which RNTuple the field belongs to, in case the field belongs to an auxiliary RNTuple, according to the order
      // in which it was specified. For chained RNTuples, this value will always be 0.
      std::size_t fNTupleIdx;

   public:
      RFieldContext(std::unique_ptr<RFieldBase> protoField, REntry::RFieldToken token, std::size_t ntupleIdx = 0)
         : fProtoField(std::move(protoField)), fToken(token), fNTupleIdx(ntupleIdx)
      {
      }

      const RFieldBase &GetProtoField() const { return *fProtoField; }
      /// Concrete pages need to be reset explicitly before the page source they belong to is destroyed.
      void ResetConcreteField() { fConcreteField.reset(); }
      void SetConcreteField() { fConcreteField = fProtoField->Clone(fProtoField->GetFieldName()); }
      bool IsAuxiliary() const { return fNTupleIdx > 0; }
   };

   std::vector<RNTupleOpenSpec> fNTuples;
   std::unique_ptr<REntry> fEntry;
   std::unique_ptr<Internal::RPageSource> fPageSource;
   // Maps the (qualified) field name to its corresponding field context.
   std::unordered_map<std::string, RFieldContext> fFieldContexts;

   NTupleSize_t fNEntriesProcessed;  //< Total number of entries processed so far
   std::size_t fCurrentNTupleNumber; //< Index of the currently open RNTuple
   NTupleSize_t fLocalEntryNumber;   //< Entry number within the current ntuple

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Creates and connects a concrete field to the current page source, based on its proto field.
   void ConnectField(RFieldContext &fieldContext, Internal::RPageSource &pageSource, REntry &entry);

   //////////////////////////////////////////////////////////////////////////
   /// \brief Advance the processor to the next available entry.
   ///
   /// \return The number of the entry loaded after advancing, or kInvalidNTupleIndex if there was no entry to advance
   /// to.
   ///
   /// Checks if the end of the currently connected RNTuple is reached. If this is the case, either the next RNTuple
   /// is connected or the iterator has reached the end.
   virtual NTupleSize_t Advance() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Fill the entry with values belonging to the current entry number.
   virtual void LoadEntry() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the local (i.e. relative to the page source currently openend) entry number. Used by
   /// `RNTupleProcessor::RIterator`.
   ///
   /// \param[in] entryNumber
   void SetLocalEntryNumber(NTupleSize_t entryNumber) { fLocalEntryNumber = entryNumber; }

   RNTupleProcessor(const std::vector<RNTupleOpenSpec> &ntuples)
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
      NTupleSize_t fCurrentEntryNumber;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = REntry;
      using difference_type = std::ptrdiff_t;
      using pointer = REntry *;
      using reference = const REntry &;

      RIterator(RNTupleProcessor &processor, NTupleSize_t entryNumber)
         : fProcessor(processor), fCurrentEntryNumber(entryNumber)
      {
         // This constructor is called with kInvalidNTupleIndex for RNTupleProcessor::end(). In that case, we already
         // know there is nothing to advance to.
         if (fCurrentEntryNumber != kInvalidNTupleIndex) {
            fProcessor.SetLocalEntryNumber(fCurrentEntryNumber);
            fCurrentEntryNumber = fProcessor.Advance();
         }
      }

      iterator operator++()
      {
         fProcessor.SetLocalEntryNumber(fCurrentEntryNumber + 1);
         fCurrentEntryNumber = fProcessor.Advance();
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         ++(*this);
         return obj;
      }

      reference operator*() { return *fProcessor.fEntry; }

      friend bool operator!=(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryNumber != rh.fCurrentEntryNumber;
      }
      friend bool operator==(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryNumber == rh.fCurrentEntryNumber;
      }
   };

   RIterator begin() { return RIterator(*this, 0); }
   RIterator end() { return RIterator(*this, kInvalidNTupleIndex); }

   static std::unique_ptr<RNTupleProcessor> Create(const RNTupleOpenSpec &ntuple);
   static std::unique_ptr<RNTupleProcessor> Create(const RNTupleOpenSpec &ntuple, RNTupleModel &model);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new RNTuple processor chain for vertical concatenation of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the ntuples to process.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first ntuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateChain(const std::vector<RNTupleOpenSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new RNTuple processor for horizontallly concatenated RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the ntuples to process. The first ntuple in the
   /// list will be considered the primary ntuple and drives the processor iteration loop. Subsequent ntuples are
   /// considered auxiliary, whose entries to be read are determined by the primary ntuple (which does not necessarily
   /// have to be sequential).
   /// \param[in] joinFields The names of the fields on which to join, in case the specified ntuples are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified RNTuple. If an empty list is provided, it is assumed that the specified ntuple are fully aligned, and
   /// `RNTupleIndex` will not be used.
   /// \param[in] models A list of models for the ntuples. This list must either contain a model for each ntuple in
   /// `ntuples` (following the specification order), or be empty. When the list is empty, the default model (i.e.
   /// containing all fields) will be used for each ntuple.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateJoin(const std::vector<RNTupleOpenSpec> &ntuples,
                                                       const std::vector<std::string> &joinFields,
                                                       std::vector<std::unique_ptr<RNTupleModel>> models = {});
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleSingleProcessor
\ingroup NTuple
\brief Processor specializiation for processing a single RNTuple.
*/
// clang-format on
class RNTupleSingleProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleProcessor for processing a single RNTuple.
   ///
   /// \param[in] ntuple The source specification (name and storage location) for the RNTuple to process.
   /// \param[in] model The model that specifies which fields should be read by the processor.
   RNTupleSingleProcessor(const RNTupleOpenSpec &ntuple, RNTupleModel &model);

   NTupleSize_t Advance() final;

public:
   void LoadEntry() { fEntry->Read(fLocalEntryNumber); }
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
   NTupleSize_t Advance() final;
   void LoadEntry() final { fEntry->Read(fLocalEntryNumber); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect an RNTuple for processing.
   ///
   /// \param[in] ntuple The RNTupleOpenSpec describing the RNTuple to connect.
   ///
   /// \return The number of entries in the newly-connected RNTuple.
   ///
   /// Creates and attaches new page source for the specified RNTuple, and connects the fields that are known by
   /// the processor to it.
   NTupleSize_t ConnectNTuple(const RNTupleOpenSpec &ntuple);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleChainProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the descriptor of the first RNTuple specified in `ntuples`.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainProcessor(const std::vector<RNTupleOpenSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleJoinProcessor
\ingroup NTuple
\brief Processor specializiation for horizontally concatenated RNTuples (joins).
*/
// clang-format on
class RNTupleJoinProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   std::unique_ptr<RNTupleModel> fJoinModel;
   std::vector<std::unique_ptr<Internal::RPageSource>> fAuxiliaryPageSources;
   /// Tokens representing the join fields present in the main RNTuple
   std::vector<REntry::RFieldToken> fJoinFieldTokens;
   std::vector<std::unique_ptr<Internal::RNTupleIndex>> fJoinIndices;

   bool IsUsingIndex() const { return fJoinIndices.size() > 0; }

   NTupleSize_t Advance() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Fill the entry with values belonging to the current entry number of the primary RNTuple.
   void LoadEntry() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Constructs a new RNTupleJoinProcessor.
   ///
   /// \param[in] mainNTuple The source specification (name and storage location) of the primary RNTuple.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the RNTuple's descriptor.
   RNTupleJoinProcessor(const RNTupleOpenSpec &mainNTuple, std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add an auxiliary RNTuple to the processor.
   ///
   /// \param[in] auxNTuple The source specification (name and storage location) of the auxiliary RNTuple.
   /// \param[in] joinFields The names of the fields used in the join.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the RNTuple's descriptor.
   void AddAuxiliary(const RNTupleOpenSpec &auxNTuple, const std::vector<std::string> &joinFields,
                     std::unique_ptr<RNTupleModel> model = nullptr);
   void ConnectFields();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Populate fJoinFieldTokens with tokens for join fields belonging to the main RNTuple in the join model.
   ///
   /// \param[in] joinFields The names of the fields used in the join.
   void SetJoinFieldTokens(const std::vector<std::string> &joinFields)
   {
      fJoinFieldTokens.reserve(joinFields.size());
      for (const auto &fieldName : joinFields) {
         fJoinFieldTokens.emplace_back(fEntry->GetToken(fieldName));
      }
   }

public:
   RNTupleJoinProcessor(const RNTupleJoinProcessor &) = delete;
   RNTupleJoinProcessor operator=(const RNTupleJoinProcessor &) = delete;
   RNTupleJoinProcessor(RNTupleJoinProcessor &&) = delete;
   RNTupleJoinProcessor operator=(RNTupleJoinProcessor &&) = delete;
   ~RNTupleJoinProcessor() override
   {
      for (auto &[_, fieldContext] : fFieldContexts) {
         fieldContext.ResetConcreteField();
      }
   }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleProcessor
