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
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleProcessorEntry.hxx>
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

// clang-format off
/**
\class ROOT::Experimental::RNTupleProcessorOptionalPtr<T>
\ingroup NTuple
\brief The RNTupleProcessorOptionalPtr provides access to values from fields present in an RNTupleProcessor, with support
and checks for missing values.
*/
// clang-format on
template <typename T>
class RNTupleProcessorOptionalPtr {
   friend class RNTupleProcessor;

private:
   Internal::RNTupleProcessorEntry *fProcessorEntry;
   Internal::RNTupleProcessorEntry::FieldIndex_t fFieldIndex;

   RNTupleProcessorOptionalPtr(Internal::RNTupleProcessorEntry *processorEntry,
                               Internal::RNTupleProcessorEntry::FieldIndex_t fieldIdx)
      : fProcessorEntry(processorEntry), fFieldIndex(fieldIdx)
   {
   }

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the pointer currently holds a valid value.
   bool HasValue() const { return fProcessorEntry->IsValidField(fFieldIndex); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a shared pointer to the field value managed by the processor's entry.
   ///
   /// \return A `std::shared_ptr<T>` if the field is valid in the current entry, or a `nullptr` otherwise.
   std::shared_ptr<T> GetPtr() const
   {
      if (fProcessorEntry->IsValidField(fFieldIndex))
         return fProcessorEntry->GetPtr<T>(fFieldIndex);

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a non-owning pointer to the field value managed by the processor's entry.
   ///
   /// \return A `T*` if the field is valid in the current entry, or a `nullptr` otherwise.
   T *GetRawPtr() const { return GetPtr().get(); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind the value to `valuePtr`.
   ///
   /// \param[in] valuePtr Pointer to bind the value to.
   ///
   /// \warning Use this function with care! Values may not always be valid for every entry during processing, for
   /// example when a field is not present in one of the chained processors or when during a join operation, no matching
   /// entry in the auxiliary processor can be found. Reading `valuePtr` as-is therefore comes with the risk of reading
   /// invalid data. After binding a pointer to an `RNTupleProcessorOptionalPtr`, we *strongly* recommend only accessing
   /// its data through this interface, to ensure that only valid data can be read.
   void BindRawPtr(T *valuePtr) { fProcessorEntry->BindRawPtr(fFieldIndex, valuePtr); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to the field value managed by the processor's entry.
   ///
   /// Throws an exception if the field is invalid in the processor's current entry.
   const T &operator*() const
   {
      if (auto ptr = GetPtr())
         return *ptr;
      else
         throw RException(R__FAIL("cannot read \"" + fProcessorEntry->FindFieldName(fFieldIndex) +
                                  "\" because it has no value for the current entry"));
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Access the field value managed by the processor's entry.
   ///
   /// Throws an exception if the field is invalid in the processor's current entry.
   const T *operator->() const
   {
      if (auto ptr = GetPtr())
         return ptr.get();
      else
         throw RException(R__FAIL("cannot read \"" + fProcessorEntry->FindFieldName(fFieldIndex) +
                                  "\" because it has no value for the current entry"));
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleProcessorOptionalPtr<void>
\ingroup NTuple
\brief Specialization of RNTupleProcessorOptionalPtr<T> for `void`-type pointers.
*/
// clang-format on
template <>
class RNTupleProcessorOptionalPtr<void> {
   friend class RNTupleProcessor;

private:
   Internal::RNTupleProcessorEntry *fProcessorEntry;
   Internal::RNTupleProcessorEntry::FieldIndex_t fFieldIndex;

   RNTupleProcessorOptionalPtr(Internal::RNTupleProcessorEntry *processorEntry,
                               Internal::RNTupleProcessorEntry::FieldIndex_t fieldIdx)
      : fProcessorEntry(processorEntry), fFieldIndex(fieldIdx)
   {
   }

public:
   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the pointer currently holds a valid value.
   bool HasValue() const { return fProcessorEntry->IsValidField(fFieldIndex); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the pointer to the field value managed by the processor's entry.
   ///
   /// \return A `std::shared_ptr<void>` if the field is valid in the current entry, or a `nullptr` otherwise.
   std::shared_ptr<void> GetPtr() const
   {
      if (fProcessorEntry->IsValidField(fFieldIndex))
         return fProcessorEntry->GetPtr<void>(fFieldIndex);

      return nullptr;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a non-owning pointer to the field value managed by the processor's entry.
   ///
   /// \return A `void*` if the field is valid in the current entry, or a `nullptr` otherwise.
   void *GetRawPtr() const { return GetPtr().get(); }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Bind the value to `valuePtr`.
   ///
   /// \param[in] valuePtr Pointer to bind the value to.
   ///
   /// \warning Use this function with care! Values may not always be valid for every entry during processing, for
   /// example when a field is not present in one of the chained processors or when during a join operation, no matching
   /// entry in the auxiliary processor can be found. Reading `valuePtr` as-is therefore comes with the risk of reading
   /// invalid data. After binding a pointer to an `RNTupleProcessorOptionalPtr`, we *strongly* recommend only accessing
   /// its data through this interface, to ensure that only valid data can be read.
   void BindRawPtr(void *valuePtr) { fProcessorEntry->BindRawPtr(fFieldIndex, valuePtr); }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleProcessor
\ingroup NTuple
\brief Interface for iterating over entries of vertically ("chained") and/or horizontally ("joined") combined RNTuples.

Example usage (see ntpl012_processor_chain.C and ntpl015_processor_join.C for bigger examples):

~~~{.cpp}
#include <ROOT/RNTupleProcessor.hxx>
using ROOT::Experimental::RNTupleProcessor;
using ROOT::Experimental::RNTupleOpenSpec;

std::vector<RNTupleOpenSpec> ntuples = {{"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}};
auto processor = RNTupleProcessor::CreateChain(ntuples);

auto pt = processor->RequestField<float>("pt");

for (const auto idx : *processor) {
   std::cout << "event = " << idx << ", pt = " << *pt << std::endl;
}
~~~

An RNTupleProcessor is created either:
1. By providing one or more RNTupleOpenSpecs, each of which contains the name and storage location of a single RNTuple;
2. By providing a previously created RNTupleProcessor.

The RNTupleProcessor provides an iterator which gives access to the index of the current *global* entry of the
processor, i.e. taking into account previously processed RNTuples.

Because the schemas of each RNTuple that are part of an RNTupleProcessor may not necessarily be identical, or because
it can occur that entries are only partially complete in a join-based processor, field values may be marked as
"invalid", at which point their data should not be read. This is handled by the RNTupleProcessorOptionalPtr
that is returned by RequestField().
*/
// clang-format on
class RNTupleProcessor {
   friend struct ROOT::Experimental::Internal::RNTupleProcessorEntryLoader; // for unit tests
   friend class RNTupleSingleProcessor;
   friend class RNTupleChainProcessor;
   friend class RNTupleJoinProcessor;

protected:
   std::string fProcessorName;
   std::unique_ptr<ROOT::RNTupleModel> fProtoModel = nullptr;
   std::shared_ptr<Internal::RNTupleProcessorEntry> fEntry = nullptr;
   std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> fFieldIdxs;

   /// Total number of entries. Only to be used internally by the processor, not meant to be exposed in the public
   /// interface.
   ROOT::NTupleSize_t fNEntries = kInvalidNTupleIndex;

   ROOT::NTupleSize_t fNEntriesProcessed = 0;  //< Total number of entries processed so far
   ROOT::NTupleSize_t fCurrentEntryNumber = 0; //< Current processor entry number
   std::size_t fCurrentProcessorNumber = 0;    //< Number of the currently open inner processor

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Initialize the processor, by setting `fProtoModel` and creating an (initially empty) `fEntry`, or setting
   /// an existing one.
   virtual void Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor already has been initialized.
   bool IsInitialized() const { return fProtoModel && fEntry; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect fields to the page source of the processor's underlying RNTuple(s).
   ///
   /// \param[in] fieldIdxs Indices of the fields to connect.
   /// \param[in] provenance Provenance of the processor.
   /// \param[in] updateFields Whether the fields in the entry need to be updated, because the current underlying
   /// RNTuple source changed.
   virtual void Connect(const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
                        const Internal::RNTupleProcessorProvenance &provenance, bool updateFields) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number.
   ///
   /// \param[in] entryNumber Entry number to load
   ///
   /// \return `entryNumber` if the entry was successfully loaded, `kInvalidNTupleIndex` otherwise.
   virtual ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the proto model used by the processor.
   ///
   /// A processor's proto model contains all fields that can be accessed and is inferred from the descriptors of the
   /// underlying RNTuples. It is used in RequestField() to check that the requested field is actually valid.
   const ROOT::RNTupleModel &GetProtoModel() const
   {
      assert(fProtoModel);
      return *fProtoModel;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor
   virtual ROOT::NTupleSize_t GetNEntries() = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \param[in] fieldName Name of the field to check.
   virtual bool CanReadFieldFromDisk(std::string_view fieldName) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   ///
   /// \param[in] fieldName Name of the field to add.
   /// \param[in] valuePtr Pointer to bind to the field's value in the entry. If this is a `nullptr`, a pointer will be
   /// created.
   /// \param[in] provenance Provenance of the processor.
   ///
   /// \return The index of the newly added field in the entry.
   ///
   /// In case the field was already present in the entry, the index of the existing field is returned.
   virtual ROOT::RResult<Internal::RNTupleProcessorEntry::FieldIndex_t>
   AddFieldToEntry(std::string_view fieldName, void *valuePtr,
                   const Internal::RNTupleProcessorProvenance &provenance) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \param[in] joinTable the join table to map the entries to.
   /// \param[in] entryOffset In case the entry mapping is added from a chain, the offset of the entry indexes to use
   /// with respect to the processor's position in the chain.
   virtual void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Processor-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \param[in,out] output Output stream to print to.
   virtual void PrintStructureImpl(std::ostream &output) const = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new base RNTupleProcessor.
   ///
   /// \param[in] processorName Name of the processor. By default, this is the name of the underlying RNTuple for
   /// RNTupleSingleProcessor, the name of the first processor for RNTupleChainProcessor, or the name of the primary
   /// RNTuple for RNTupleJoinProcessor.
   RNTupleProcessor(std::string_view processorName) : fProcessorName(processorName) {}

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
   /// primary processor for RNTupleJoinProcessor.
   const std::string &GetProcessorName() const { return fProcessorName; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Request access to a field for reading during processing.
   ///
   /// \tparam T Type of the requested field.
   ///
   /// \param[in] fieldName Name of the requested field.
   ///
   /// \return An RNTupleProcessorOptionalPtr, which provides access to the field's value.
   ///
   /// \warning Provide a `valuePtr` with care! Values may not always be valid for every entry during processing, for
   /// example when a field is not present in one of the chained processors or when during a join operation, no matching
   /// entry in the auxiliary processor can be found. Reading `valuePtr` as-is therefore comes with the risk of reading
   /// invalid data. After passing a pointer to `RequestField`, we *strongly* recommend only accessing its data through
   /// the interface of the returned `RNTupleProcessorOptionalPtr`, to ensure that only valid data can be read.
   template <typename T>
   RNTupleProcessorOptionalPtr<T> RequestField(std::string_view fieldName, void *valuePtr = nullptr)
   {
      Initialize(fEntry);
      // TODO handle alternative (compatible field types)
      auto fieldIdx = AddFieldToEntry(fieldName, valuePtr, Internal::RNTupleProcessorProvenance()).Unwrap();
      return RNTupleProcessorOptionalPtr<T>(fEntry.get(), fieldIdx);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Print a graphical representation of the processor composition.
   ///
   /// \param[in,out] output Stream to print to (default is stdout).
   ///
   /// ### Example:
   /// The structure of a processor representing a join between a single primary RNTuple and a chain of two auxiliary
   /// RNTuples will be printed as follows:
   /// ~~~
   /// +-----------------------------+ +-----------------------------+
   /// | ntuple                      | | ntuple_aux                  |
   /// | ntuple.root                 | | ntuple_aux1.root            |
   /// +-----------------------------+ +-----------------------------+
   ///                                 +-----------------------------+
   ///                                 | ntuple_aux                  |
   ///                                 | ntuple_aux2.root            |
   ///                                 +-----------------------------+
   /// ~~~
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
      ROOT::NTupleSize_t fCurrentEntryNumber;

   public:
      using iterator_category = std::input_iterator_tag;
      using iterator = RIterator;
      using value_type = ROOT::NTupleSize_t;
      using difference_type = std::ptrdiff_t;
      using pointer = ROOT::NTupleSize_t *;
      using reference = ROOT::NTupleSize_t &;

      RIterator(RNTupleProcessor &processor, ROOT::NTupleSize_t entryNumber)
         : fProcessor(processor), fCurrentEntryNumber(entryNumber)
      {
         if (!fProcessor.fEntry) {
            fCurrentEntryNumber = ROOT::kInvalidNTupleIndex;
         }
         // This constructor is called with kInvalidNTupleIndex for RNTupleProcessor::end(). In that case, we already
         // know there is nothing to load.
         if (fCurrentEntryNumber != ROOT::kInvalidNTupleIndex) {
            fProcessor.Connect(fProcessor.fEntry->GetFieldIndices(), Internal::RNTupleProcessorProvenance(),
                               /*updateFields=*/false);
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

      reference operator*() { return fCurrentEntryNumber; }

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
   /// \brief Initialize the processor, by setting `fProtoModel` and creating an (initially empty) `fEntry`, or setting
   /// an existing one.
   ///
   /// At this point, the page source for the underlying RNTuple of the processor will be created and opened.
   void Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   void Connect(const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final
   {
      Initialize();
      if (fNEntries == ROOT::kInvalidNTupleIndex)
         Connect(fFieldIdxs);
      return fNEntries;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \sa RNTupleProcessor::CanReadFieldFromDisk()
   bool CanReadFieldFromDisk(std::string_view fieldName) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   /// \sa RNTupleProcessor::AddFieldToEntry()
   ROOT::RResult<Internal::RNTupleProcessorEntry::FieldIndex_t> AddFieldToEntry(
      std::string_view fieldName, void *valuePtr = nullptr,
      const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Processor-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::PrintStructureImpl
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
      // The proto model needs to be deleted before fPageSource.
      fProtoModel.release();
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
   std::vector<ROOT::NTupleSize_t> fInnerNEntries;

   Internal::RNTupleProcessorProvenance fProvenance;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Initialize the processor, by setting `fProtoModel` and creating an (initially empty) `fEntry`, or setting
   /// an existing one.
   void Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   ///
   /// \sa RNTupleProcessor::Connect()
   void Connect(const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the entry to reflect any missing fields in the current inner processor.
   void ConnectInnerProcessor(std::size_t processorNumber);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ///
   /// \note This requires opening all underlying RNTuples being processed in the chain, and could become costly!
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \sa RNTupleProcessor::CanReadFieldFromDisk()
   bool CanReadFieldFromDisk(std::string_view fieldName) final
   {
      return fInnerProcessors[fCurrentProcessorNumber]->CanReadFieldFromDisk(fieldName);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   /// \sa RNTupleProcessor::AddFieldToEntry()
   ROOT::RResult<Internal::RNTupleProcessorEntry::FieldIndex_t> AddFieldToEntry(
      std::string_view fieldName, void *valuePtr = nullptr,
      const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Processor-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::PrintStructureImpl
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
   ~RNTupleChainProcessor() override = default;
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

   std::vector<std::string> fJoinFieldNames;
   std::set<Internal::RNTupleProcessorEntry::FieldIndex_t> fJoinFieldIdxs;

   std::unique_ptr<Internal::RNTupleJoinTable> fJoinTable;
   bool fJoinTableIsBuilt = false;

   std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> fAuxiliaryFieldIdxs;

   /// \brief Initialize the processor, by setting `fProtoModel` and creating an (initially empty) `fEntry`, or setting
   /// an existing one.
   void Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   ///
   /// \sa RNTupleProcessor::Connect()
   void Connect(const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number of the primary processor.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the processor's proto model by combining the primary and auxiliary models.
   ///
   /// \param[in] primaryModel The proto model of the primary processor.
   /// \param[in] auxModel The proto model of the auxiliary processors.
   ///
   /// To prevent field name clashes when one or more models have fields with duplicate names, fields from each
   /// auxiliary model are stored as a anonymous record, and subsequently registered as subfields in the join model.
   /// This way, they can be accessed from the processor's entry as `auxNTupleName.fieldName`.
   void SetProtoModel(std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::unique_ptr<ROOT::RNTupleModel> auxModel);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity for all fields in the auxiliary processor at once.
   void SetAuxiliaryFieldValidity(bool validity);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \sa RNTupleProcessor::CanReadFieldFromDisk()
   bool CanReadFieldFromDisk(std::string_view fieldName) final
   {
      if (!fPrimaryProcessor->CanReadFieldFromDisk(fieldName)) {
         if (fieldName.find(fAuxiliaryProcessor->GetProcessorName()) == 0)
            fieldName = fieldName.substr(fAuxiliaryProcessor->GetProcessorName().size() + 1);
         return fAuxiliaryProcessor->CanReadFieldFromDisk(fieldName);
      }

      return true;
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   /// \sa RNTupleProcessor::AddFieldToEntry()
   ROOT::RResult<Internal::RNTupleProcessorEntry::FieldIndex_t> AddFieldToEntry(
      std::string_view fieldName, void *valuePtr = nullptr,
      const Internal::RNTupleProcessorProvenance &provenance = Internal::RNTupleProcessorProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Processor-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::PrintStructureImpl
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
   ~RNTupleJoinProcessor() override = default;
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessor
