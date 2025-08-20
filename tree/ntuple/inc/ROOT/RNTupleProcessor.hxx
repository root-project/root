/// \file ROOT/RNTupleProcessor.hxx
/// \ingroup NTuple
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

#ifndef ROOT_RNTupleProcessor
#define ROOT_RNTupleProcessor

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleProcessorEntry.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleOpenSpec
\ingroup NTuple
\brief Specification of the name and location of an RNTuple, used for creating a new RNTupleProcessor.

An RNTupleOpenSpec can be created by providing either a string with a path to the ROOT file or a pointer to the
TDirectory (or any of its subclasses) that contains the RNTuple.

Note that the RNTupleOpenSpec is *write-only*, to prevent usability issues with Python.
*/
// clang-format on
class RNTupleOpenSpec {
   friend class RNTupleProcessor;
   friend class RNTupleSingleProcessor;
   friend class RNTupleJoinProcessor;

private:
   std::string fNTupleName;
   std::variant<std::string, TDirectory *> fStorage;

public:
   RNTupleOpenSpec(std::string_view n, TDirectory *s) : fNTupleName(n), fStorage(s) {}
   RNTupleOpenSpec(std::string_view n, const std::string &s) : fNTupleName(n), fStorage(s) {}

   std::unique_ptr<ROOT::Internal::RPageSource> CreatePageSource() const;
};

enum class ENTupleProcessorKind {
   kUndefined,
   kSingle,
   kChain,
   kJoin
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
   friend class RNTupleSingleProcessor;
   friend class RNTupleChainProcessor;
   friend class RNTupleJoinProcessor;
   friend ROOT::NTupleSize_t ROOT::Experimental::Internal::GetRelativeProcessorEntryIndex(RNTupleProcessor &,
                                                                                          std::string_view,
                                                                                          ROOT::NTupleSize_t);

protected:
   static inline ENTupleProcessorKind fKind;
   std::string fProcessorName;
   std::shared_ptr<Internal::RNTupleProcessorEntry> fEntry = nullptr;

   /// Total number of entries. Only to be used internally by the processor, not meant to be exposed in the public
   /// interface.
   ROOT::NTupleSize_t fNEntries = kInvalidNTupleIndex;

   ROOT::NTupleSize_t fCurrentEntryIndex = kInvalidNTupleIndex; //< Current processor entry index

   static ENTupleProcessorKind GetKind() { return fKind; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect a new (nested) processor.
   ///
   /// \param[in] entry Optional entry passed by the parent processor. When no entry is provided, a new, empty entry
   /// will be created.
   virtual void Connect(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the internal processor bookkeeping so the proper entry values can be loaded.
   ///
   /// \param[in] globalIndex Global entry index to read.
   ///
   /// Update() only updates the bookkeeping information internal to the processor, but does not read any data. This is
   /// done using views (i.e., RNTupleProcessor::GetView()).
   virtual void Update(ROOT::NTupleSize_t globalIndex) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor has an entry following fCurrentEntryIndex.
   virtual bool HasNextEntry() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field can exists in the processor's schema.
   ///
   /// \param[in] fieldName Name of the field to check.
   virtual bool HasField(std::string_view fieldName) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new field for processing.
   ///
   /// \param[in] fieldName Name of the field to create (must exist on-disk).
   /// \param[in] typeName Type name for the field to create.
   ///
   /// \return The newly created field.
   virtual std::unique_ptr<ROOT::RFieldBase>
   CreateField(std::string_view fieldName, std::string_view typeName = "") = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the processor's entry.
   ///
   /// \param[in] fieldName Canonical name of the field to add. This might be different from the actual on-disk field
   /// name in case the field is nested.
   /// \param[in] field Field to add.
   ///
   /// \return A reference to the RNTupleProcessorValue to read the field's data from the entry.
   virtual Internal::RNTupleProcessorValue &
   AddFieldToEntry(std::string_view fieldName, std::unique_ptr<RFieldBase> field)
   {
      assert(fEntry);
      return fEntry->AddOrGetValue(fieldName, std::move(field));
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ///
   /// \warning This method requires connecting every nested processor and can potentially incur a large overhead if not
   /// used with care.
   virtual ROOT::NTupleSize_t GetNEntries() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry index relative to `fCurrentEntryIndex` in innermost processor for the given field, so its
   /// value can be read from disk.
   ///
   /// \param[in] fieldName Name of the field for which to get the local entry index.
   ///
   /// \return Index local to the processor for the given field, or ROOT::kInvalidEntryIndex if it doesn't exist.
   virtual ROOT::NTupleSize_t GetLocalCurrentEntryIndex(std::string_view fieldName) const = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \param[in] joinTable the join table to map the entries to.
   /// \param[in] entryOffset In case the entry mapping is added from a chain, the offset of the entry indexes to use
   /// with respect to the processor's position in the chain.
   virtual void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) = 0;

   virtual void PrintStructureImpl(std::ostream &output) const = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new base RNTupleProcessor.
   ///
   /// \param[in] kind The kind of processor.
   /// \param[in] processorName Name of the processor. By default, this is the name of the underlying RNTuple for
   /// RNTupleSingleProcessor, the name of the first processor for RNTupleChainProcessor, or the name of the primary
   /// RNTuple for RNTupleJoinProcessor.
   RNTupleProcessor(ENTupleProcessorKind kind, std::string_view processorName) : fProcessorName(processorName)
   {
      fKind = kind;
   }

public:
   RNTupleProcessor(const RNTupleProcessor &) = delete;
   RNTupleProcessor(RNTupleProcessor &&) = delete;
   RNTupleProcessor &operator=(const RNTupleProcessor &) = delete;
   RNTupleProcessor &operator=(RNTupleProcessor &&) = delete;
   virtual ~RNTupleProcessor() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry index that is currently being processed.
   ROOT::NTupleSize_t GetCurrentEntryIndex() const { return fCurrentEntryIndex; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of the processor.
   ///
   /// Unless this name was explicitly specified during creation of the processor, this is the name of the underlying
   /// RNTuple for RNTupleSingleProcessor, the name of the first processor for RNTupleChainProcessor, or the name of the
   /// primary processor for RNTupleJoinProcessor.
   const std::string &GetProcessorName() const { return fProcessorName; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get access to an individual field in the processor.
   ///
   /// \tparam T Type to read the field values into.
   ///
   /// \param[in] fieldName Name of the field to create the view for
   ///
   /// \return An RNTupleProcessorView of type T for the provided field.
   template <typename T>
   RNTupleProcessorView<T> GetView(std::string_view fieldName)
   {
      if (!fEntry)
         Connect();
      auto typeName = ROOT::Internal::GetRenormalizedTypeName(typeid(T));
      auto field = CreateField(fieldName, typeName);
      if (!field)
         throw RException(R__FAIL("could not create view for field \"" + std::string(fieldName) + "\""));
      auto &value = AddFieldToEntry(fieldName, std::move(field));
      return RNTupleProcessorView<T>(*this, value);
   }

   void PrintStructure(std::ostream &output = std::cout) { PrintStructureImpl(output); }

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
      ROOT::NTupleSize_t fCurrentEntryIndex;

   public:
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = ROOT::NTupleSize_t;
      using difference_type = std::ptrdiff_t;
      using pointer = ROOT::NTupleSize_t *;
      using reference = ROOT::NTupleSize_t &;

      RIterator(RNTupleProcessor &processor, ROOT::NTupleSize_t entryNumber)
         : fProcessor(processor), fCurrentEntryIndex(entryNumber)
      {
         if (fCurrentEntryIndex != kInvalidNTupleIndex) {
            if (!fProcessor.fEntry)
               fProcessor.Connect();
            fProcessor.Update(fCurrentEntryIndex);

            if (!fProcessor.HasNextEntry())
               fCurrentEntryIndex = ROOT::kInvalidNTupleIndex;
         }
      }

      iterator operator++()
      {
         if (fProcessor.HasNextEntry()) {
            fProcessor.Update(++fCurrentEntryIndex);
         } else {
            fCurrentEntryIndex = kInvalidNTupleIndex;
         }
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         ++(*this);
         return obj;
      }

      reference operator*() { return fCurrentEntryIndex; }

      friend bool operator!=(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryIndex != rh.fCurrentEntryIndex;
      }
      friend bool operator==(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryIndex == rh.fCurrentEntryIndex;
      }
   };

   RIterator begin() { return RIterator(*this, 0); }
   RIterator end() { return RIterator(*this, ROOT::kInvalidNTupleIndex); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a single RNTuple.
   ///
   /// \param[in] ntuple The name and storage location of the RNTuple to process.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the input RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> Create(RNTupleOpenSpec ntuple, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the RNTuples to process.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of other RNTupleProcessors.
   ///
   /// \param[in] innerProcessors A list with the processors to chain.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first inner processor is
   /// used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *join* (i.e., a horizontal combination) of RNTuples.
   ///
   /// \param[in] primaryNTuple The name and location of the primary RNTuple. Its entries are processed in sequential
   /// order.
   /// \param[in] auxNTuple The name and location of the RNTuple to join the primary RNTuple with. The order in which
   /// its entries are processed is determined by the primary RNTuple and doesn't necessarily have to be sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified RNTuples are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified RNTuple. If an empty list is provided, it is assumed that the specified ntuple are fully aligned.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the primary RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                       const std::vector<std::string> &joinFields,
                                                       std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *join* (i.e., a horizontal combination) of RNTuples.
   ///
   /// \param[in] primaryProcessor The primary processor. Its entries are processed in sequential order.
   /// \param[in] auxProcessor The processor to join the primary processor with. The order in which its entries are
   /// processed is determined by the primary processor and doesn't necessarily have to be sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified processors are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified processors. If an empty list is provided, it is assumed that the specified processors are fully
   /// aligned.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the primary processor is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateJoin(std::unique_ptr<RNTupleProcessor> primaryProcessor, std::unique_ptr<RNTupleProcessor> auxProcessor,
              const std::vector<std::string> &joinFields, std::string_view processorName = "");
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
   std::unique_ptr<ROOT::Internal::RPageSource> fPageSource;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the page source of the underlying RNTuple.
   ///
   /// \sa RNTupleProcessor::Connect()
   void Connect(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the internal processor bookkeeping so the proper entry values can be loaded.
   ///
   /// \sa RNTupleProcessor::Update()
   void Update(ROOT::NTupleSize_t globalIndex) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor has an entry following fCurrentEntryIndex.
   bool HasNextEntry() final { return fNEntries != 0 && fCurrentEntryIndex < fNEntries - 1; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field can exists in the processor's schema.
   ///
   /// \sa RNTupleProcessor::HasField;
   bool HasField(std::string_view fieldName) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new field for processing.
   ///
   /// \sa RNTupleProcessor::CreateField
   std::unique_ptr<ROOT::RFieldBase> CreateField(std::string_view fieldName, std::string_view typeName = "") final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final
   {
      Connect();
      return fNEntries;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry index relative to `fCurrentEntryIndex` in innermost processor for the given field, so its
   /// value can be read from disk.
   ///
   /// \sa RNTupleProcessor::GetLocalCurrentEntryIndex
   ROOT::NTupleSize_t GetLocalCurrentEntryIndex(std::string_view /* fieldName */) const final
   {
      return fCurrentEntryIndex;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleProcessor for processing a single RNTuple.
   ///
   /// \param[in] ntuple The source specification (name and storage location) for the RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::Create, this is
   /// the name of the underlying RNTuple.
   RNTupleSingleProcessor(RNTupleOpenSpec ntuple, std::string_view processorName);

public:
   RNTupleSingleProcessor(const RNTupleSingleProcessor &) = delete;
   RNTupleSingleProcessor(RNTupleSingleProcessor &&) = delete;
   RNTupleSingleProcessor &operator=(const RNTupleSingleProcessor &) = delete;
   RNTupleSingleProcessor &operator=(RNTupleSingleProcessor &&) = delete;
   ~RNTupleSingleProcessor() override
   {
      if (fEntry)
         fEntry->Clear();
   };
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
   std::size_t fCurrentProcessorIndex = 0;         //< Number of the currently open inner processor
   std::vector<ROOT::NTupleSize_t> fInnerNEntries; //< Number of entries in each inner processor

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the processor and the first nested processor in the chain.
   ///
   /// \sa RNTupleProcessor::Connect()
   void Connect(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the internal processor bookkeeping so the proper entry values can be loaded.
   ///
   /// \sa RNTupleProcessor::Update()
   void Update(ROOT::NTupleSize_t globalIndex) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor has an entry following fCurrentEntryIndex.
   bool HasNextEntry() final
   {
      return fInnerProcessors[fCurrentProcessorIndex]->HasNextEntry() ||
             fCurrentProcessorIndex < fInnerProcessors.size() - 1;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field can exists in the processor's schema.
   ///
   /// \sa RNTupleProcessor::HasField;
   bool HasField(std::string_view fieldName) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new field for processing.
   ///
   /// \sa RNTupleProcessor::CreateField
   std::unique_ptr<ROOT::RFieldBase> CreateField(std::string_view fieldName, std::string_view typeName = "") final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry index relative to `fCurrentEntryIndex` in innermost processor for the given field, so its
   /// value can be read from disk.
   ///
   /// \sa RNTupleProcessor::GetLocalCurrentEntryIndex
   ROOT::NTupleSize_t GetLocalCurrentEntryIndex(std::string_view fieldName) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleChainProcessor.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateChain, this
   /// is the name of the first inner processor.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainProcessor(std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName);

public:
   RNTupleChainProcessor(const RNTupleChainProcessor &) = delete;
   RNTupleChainProcessor(RNTupleChainProcessor &&) = delete;
   RNTupleChainProcessor &operator=(const RNTupleChainProcessor &) = delete;
   RNTupleChainProcessor &operator=(RNTupleChainProcessor &&) = delete;
   ~RNTupleChainProcessor() override
   {
      if (fEntry)
         fEntry->Clear();
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleJoinProcessor
\ingroup NTuple
\brief Processor specialization for horizontally combined (*joined*) RNTupleProcessors.
*/
// clang-format on
class RNTupleJoinProcessor : public RNTupleProcessor {
   friend class RNTupleProcessor;

private:
   std::unique_ptr<RNTupleProcessor> fPrimaryProcessor;
   std::unique_ptr<RNTupleProcessor> fAuxiliaryProcessor;

   std::shared_ptr<Internal::RNTupleProcessorEntry> fAuxiliaryEntry;

   /// Names of the join fields present in the primary processor.
   std::vector<std::string> fJoinFields;
   std::unique_ptr<Internal::RNTupleJoinTable> fJoinTable;
   bool fJoinTableIsBuilt = false;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the processor and the first nested processor in the chain.
   ///
   /// \sa RNTupleProcessor::Connect()
   void Connect(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the internal processor bookkeeping so the proper entry values can be loaded.
   ///
   /// \sa RNTupleProcessor::Update()
   void Update(ROOT::NTupleSize_t globalIndex) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor has an entry following fCurrentEntryIndex.
   bool HasNextEntry() final { return fPrimaryProcessor->HasNextEntry(); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check whether a field can exists in the processor's schema.
   ///
   /// \sa RNTupleProcessor::HasField;
   bool HasField(std::string_view fieldName) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new field for processing.
   ///
   /// \sa RNTupleProcessor::CreateField
   std::unique_ptr<ROOT::RFieldBase> CreateField(std::string_view fieldName, std::string_view typeName = "") final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new field to the processor's entry.
   ///
   /// \sa RNTupleProcessor::AddFieldToEntry
   Internal::RNTupleProcessorValue &
   AddFieldToEntry(std::string_view fieldName, std::unique_ptr<RFieldBase> field) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry index relative to `fCurrentEntryIndex` in innermost processor for the given field, so its
   /// value can be read from disk.
   ///
   /// \sa RNTupleProcessor::GetLocalCurrentEntryIndex
   ROOT::NTupleSize_t GetLocalCurrentEntryIndex(std::string_view fieldName) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleJoinProcessor.
   /// \param[in] primaryProcessor The primary processor. Its entries are processed in sequential order.
   /// \param[in] auxProcessor The processor to join the primary processor with. The order in which its entries are
   /// processed is determined by the primary processor and doesn't necessarily have to be sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified processors are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified processor. If an empty list is provided, it is assumed that the processors are fully aligned.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateJoin, this
   /// is the name of the primary processor.
   RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                        std::unique_ptr<RNTupleProcessor> auxProcessor, const std::vector<std::string> &joinFields,
                        std::string_view processorName);

public:
   RNTupleJoinProcessor(const RNTupleJoinProcessor &) = delete;
   RNTupleJoinProcessor operator=(const RNTupleJoinProcessor &) = delete;
   RNTupleJoinProcessor(RNTupleJoinProcessor &&) = delete;
   RNTupleJoinProcessor operator=(RNTupleJoinProcessor &&) = delete;
   ~RNTupleJoinProcessor() override
   {
      if (fEntry)
         fEntry->Clear();
      if (fAuxiliaryEntry)
         fAuxiliaryEntry->Clear();
   }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessor
