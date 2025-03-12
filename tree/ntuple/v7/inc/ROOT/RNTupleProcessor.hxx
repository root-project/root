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
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Internal {
struct RNTupleProcessorEntryLoader;
} // namespace Internal

/// Used to specify the underlying RNTuples in RNTupleProcessor
struct RNTupleOpenSpec {
   std::string fNTupleName;
   std::string fStorage;

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
   friend struct ROOT::Experimental::Internal::RNTupleProcessorEntryLoader; // for unit tests
   friend class RNTupleSingleProcessor;
   friend class RNTupleChainProcessor;
   friend class RNTupleJoinProcessor;

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

   std::string fProcessorName;
   std::vector<RNTupleOpenSpec> fNTuples;
   std::unique_ptr<REntry> fEntry;
   std::unique_ptr<Internal::RPageSource> fPageSource;
   /// Maps the (qualified) field name to its corresponding field context.
   std::unordered_map<std::string, RFieldContext> fFieldContexts;

   std::unique_ptr<RNTupleModel> fModel;

   /// Total number of entries. Only to be used internally by the processor, not meant to be exposed in the public
   /// interface.
   ROOT::NTupleSize_t fNEntries = kInvalidNTupleIndex;

   ROOT::NTupleSize_t fNEntriesProcessed = 0;  //< Total number of entries processed so far
   ROOT::NTupleSize_t fCurrentEntryNumber = 0; //< Current processor entry number
   std::size_t fCurrentProcessorNumber = 0;    //< Number of the currently open inner processor

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create and connect a concrete field to the current page source, based on its proto field.
   void ConnectField(RFieldContext &fieldContext, Internal::RPageSource &pageSource, REntry &entry);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number.
   ///
   /// \param[in] entryNumber Entry number to load
   ///
   /// \return `entryNumber` if the entry was successfully loaded, `kInvalidNTupleIndex` otherwise.
   virtual ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Point the entry's field values of the processor to the pointers from the provided entry.
   ///
   /// \param[in] entry The entry whose field values to use.
   virtual void SetEntryPointers(const REntry &entry) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor
   virtual ROOT::NTupleSize_t GetNEntries() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new base RNTupleProcessor.
   ///
   /// \param[in] processorName Name of the processor. By default, this is the name of the underlying RNTuple for
   /// RNTupleSingleProcessor, the name of the first processor for RNTupleChainProcessor, or the name of the primary
   /// RNTuple for RNTupleJoinProcessor.
   /// \param[in] model The RNTupleModel representing the entries returned by the processor.
   ///
   /// \note Before processing, a model *must* exist. However, this is handled downstream by the RNTupleProcessor's
   /// factory functions (CreateSingle, CreateChain and CreateJoin) and constructors.
   RNTupleProcessor(std::string_view processorName, std::unique_ptr<RNTupleModel> model)
      : fProcessorName(processorName), fModel(std::move(model))
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
   ROOT::NTupleSize_t GetNEntriesProcessed() const { return fNEntriesProcessed; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number that is currently being processed.
   ROOT::NTupleSize_t GetCurrentEntryNumber() const { return fCurrentEntryNumber; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of the inner processor currently being read.
   ///
   /// This method is only relevant for the RNTupleChainProcessor. For the other processors, 0 is always returned.
   std::size_t GetCurrentProcessorNumber() const { return fCurrentProcessorNumber; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of the processor.
   ///
   /// Unless this name was explicitly specified during creation of the processor, this is the name of the underlying
   /// RNTuple for RNTupleSingleProcessor, the name of the first processor for RNTupleChainProcessor, or the name of the
   /// primary RNTuple for RNTupleJoinProcessor.
   const std::string &GetProcessorName() const { return fProcessorName; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the model used by the processor.
   const RNTupleModel &GetModel() const { return *fModel; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to the entry used by the processor.
   ///
   /// \return A reference to the entry used by the processor.
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
      ROOT::NTupleSize_t fCurrentEntryNumber;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = REntry;
      using difference_type = std::ptrdiff_t;
      using pointer = REntry *;
      using reference = const REntry &;

      RIterator(RNTupleProcessor &processor, ROOT::NTupleSize_t entryNumber)
         : fProcessor(processor), fCurrentEntryNumber(entryNumber)
      {
         // This constructor is called with kInvalidNTupleIndex for RNTupleProcessor::end(). In that case, we already
         // know there is nothing to load.
         if (fCurrentEntryNumber != ROOT::kInvalidNTupleIndex) {
            fCurrentEntryNumber = fProcessor.LoadEntry(fCurrentEntryNumber);
         }
      }

      iterator operator++()
      {
         fCurrentEntryNumber = fProcessor.LoadEntry(fCurrentEntryNumber + 1);
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         ++(*this);
         return obj;
      }

      reference operator*() { return fProcessor.GetEntry(); }

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
   RIterator end() { return RIterator(*this, ROOT::kInvalidNTupleIndex); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a single RNTuple.
   ///
   /// \param[in] ntuple The name and storage location of the RNTuple to process.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first ntuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   Create(const RNTupleOpenSpec &ntuple, std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a single RNTuple.
   ///
   /// \param[in] ntuple The name and storage location of the RNTuple to process.
   /// \param[in] processorName The name to give to the processor. Use
   /// Create(const RNTupleOpenSpec &, std::unique_ptr<RNTupleModel>) to automatically use the name of the input RNTuple
   /// instead.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first ntuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   Create(const RNTupleOpenSpec &ntuple, std::string_view processorName, std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the RNTuples to process.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first RNTuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateChain(const std::vector<RNTupleOpenSpec> &ntuples, std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the RNTuples to process.
   /// \param[in] processorName The name to give to the processor. Use
   /// CreateChain(const RNTupleOpenSpec &, std::unique_ptr<RNTupleModel>) to automatically use the name of the first
   /// input RNTuple instead.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first RNTuple specified.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateChain(const std::vector<RNTupleOpenSpec> &ntuples,
                                                        std::string_view processorName,
                                                        std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of other RNTupleProcessors.
   ///
   /// \param[in] innerProcessors A list with the processors to chain.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the model used by the first inner processor.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                        std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of other RNTupleProcessors.
   ///
   /// \param[in] innerProcessors A list with the processors to chain.
   /// \param[in] processorName The name to give to the processor. Use
   /// CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>>, std::unique_ptr<RNTupleModel>) to automatically use
   /// the name of the first inner processor instead.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the model used by the first inner processor.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                        std::string_view processorName,
                                                        std::unique_ptr<RNTupleModel> model = nullptr);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *join* (i.e., a horizontal combination) of RNTuples.
   ///
   /// \param[in] primaryNTuple The name and location of the primary RNTuple. Its entries are processed in sequential
   /// order.
   /// \param[in] auxNTuples The names and locations of the RNTuples to join the primary RNTuple with. The order in
   /// which their entries are processed are determined by the primary RNTuple and doesn't necessarily have to be
   /// sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified RNTuples are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified RNTuple. If an empty list is provided, it is assumed that the specified ntuple are fully aligned.
   /// \param[in] models A list of models for the RNTuples. This list must either contain a model for the primary
   /// RNTuple and each auxiliary RNTuple (following the specification order), or be empty. When the list is empty, the
   /// default model (i.e. containing all fields) will be used for each RNTuple.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateJoin(const RNTupleOpenSpec &primaryNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples,
              const std::vector<std::string> &joinFields, std::vector<std::unique_ptr<RNTupleModel>> models = {});

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *join* (i.e., a horizontal combination) of RNTuples.
   ///
   /// \param[in] primaryNTuple The name and location of the primary RNTuple. Its entries are processed in sequential
   /// order.
   /// \param[in] auxNTuples The names and locations of the RNTuples to join the primary RNTuple with. The order in
   /// which their entries are processed are determined by the primary RNTuple and doesn't necessarily have to be
   /// sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified RNTuples are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified RNTuple. If an empty list is provided, it is assumed that the specified RNTuple are fully aligned.
   /// \param[in] processorName The name to give to the processor. Use
   /// CreateJoin(const RNTupleOpenSpec &, const std::vector<RNTupleOpenSpec> &, const std::vector<std::string> &,
   /// std::vector<std::unique_ptr<RNTupleModel>>) to automatically use the name of the input RNTuple instead.
   /// \param[in] models A list of models for the RNTuples. This list must either contain a model for the primary
   /// RNTuple and each auxiliary RNTuple (following the specification order), or be empty. When the list is empty, the
   /// default model (i.e. containing all fields) will be used for each RNTuple.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateJoin(const RNTupleOpenSpec &primaryNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples,
              const std::vector<std::string> &joinFields, std::string_view processorName,
              std::vector<std::unique_ptr<RNTupleModel>> models = {});
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleSingleProcessor
\ingroup NTuple
\brief Processor specialization for processing a single RNTuple.
*/
// clang-format on
class RNTupleSingleProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   RNTupleOpenSpec fNTupleSpec;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the page source of the underlying RNTuple.
   void Connect();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \sa ROOT::Experimental::RNTupleProcessor::SetEntryPointers.
   void SetEntryPointers(const REntry &entry) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final
   {
      Connect();
      return fNEntries;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleProcessor for processing a single RNTuple.
   ///
   /// \param[in] ntuple The source specification (name and storage location) for the RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::Create, this is
   /// the name of the underlying RNTuple.
   /// \param[in] model The model that specifies which fields should be read by the processor.
   RNTupleSingleProcessor(const RNTupleOpenSpec &ntuple, std::string_view processorName,
                          std::unique_ptr<RNTupleModel> model);
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleChainProcessor
\ingroup NTuple
\brief Processor specialization for vertically combined (*chained*) RNTupleProcessors.
*/
// clang-format on
class RNTupleChainProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   std::vector<std::unique_ptr<RNTupleProcessor>> fInnerProcessors;
   std::vector<ROOT::NTupleSize_t> fInnerNEntries;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \sa ROOT::Experimental::RNTupleProcessor::SetEntryPointers.
   void SetEntryPointers(const REntry &) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ///
   /// \note This requires opening all underlying RNTuples being processed in the chain, and could become costly!
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleChainProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateChain, this
   /// is the name of the first inner processor.
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the descriptor of the first RNTuple specified in `ntuples`.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainProcessor(std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName,
                         std::unique_ptr<RNTupleModel> model);
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleJoinProcessor
\ingroup NTuple
\brief Processor specialization for horizontally combined (*joined*) RNTuples.
*/
// clang-format on
class RNTupleJoinProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   std::vector<std::unique_ptr<Internal::RPageSource>> fAuxiliaryPageSources;
   /// Tokens representing the join fields present in the main RNTuple
   std::vector<REntry::RFieldToken> fJoinFieldTokens;
   std::vector<std::unique_ptr<Internal::RNTupleJoinTable>> fJoinTables;

   bool HasJoinTable() const { return fJoinTables.size() > 0; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number of the primary RNTuple.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \sa ROOT::Experimental::RNTupleProcessor::SetEntryPointers.
   void SetEntryPointers(const REntry &) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final { return fNEntries; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleJoinProcessor.
   ///
   /// \param[in] mainNTuple The source specification (name and storage location) of the primary RNTuple.
   /// \param[in] auxNTUples The source specifications (name and storage location) of the auxiliary RNTuples.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified RNTuples are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified RNTuple. If an empty list is provided, it is assumed that the RNTuples are fully aligned.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateJoin, this
   /// is the name of the main RNTuple.
   /// \param[in] models The models that specify which fields should be read by the processor, ordered according to
   /// {mainNTuple, auxNTuple[0], ...}. The pointer returned by RNTupleModel::MakeField can be used to access a field's
   /// value during the processor iteration. When an empty list is passed, the models are created from the descriptor of
   /// each RNTuple specified in `mainNTuple` and `auxNTuple`.
   RNTupleJoinProcessor(const RNTupleOpenSpec &mainNTuple, const std::vector<RNTupleOpenSpec> &auxNTuples,
                        const std::vector<std::string> &joinFields, std::string_view processorName,
                        std::vector<std::unique_ptr<RNTupleModel>> models = {});

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

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect all fields, once the primary and all auxiliary RNTuples have been added.
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
