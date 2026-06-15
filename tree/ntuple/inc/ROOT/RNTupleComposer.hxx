/// \file ROOT/RNTupleComposer.hxx
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

#ifndef ROOT_RNTupleComposer
#define ROOT_RNTupleComposer

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleComposerEntry.hxx>
#include <ROOT/RPageStorage.hxx>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Internal {
struct RNTupleComposerEntryLoader;
} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleOpenSpec
\ingroup NTuple
\brief Specification of the name and location of an RNTuple, used for creating a new RNTupleComposer.

An RNTupleOpenSpec can be created by providing either a string with a path to the ROOT file or a pointer to the
TDirectory (or any of its subclasses) that contains the RNTuple.

Note that the RNTupleOpenSpec is *write-only*, to prevent usability issues with Python.
*/
// clang-format on
class RNTupleOpenSpec {
   friend class RNTupleComposer;
   friend class RNTupleSingleComposer;
   friend class RNTupleJoinComposer;

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
\brief The RNTupleProcessorOptionalPtr provides access to values from fields present in an RNTupleComposer, with support
and checks for missing values.
*/
// clang-format on
template <typename T>
class RNTupleProcessorOptionalPtr {
   friend class RNTupleComposer;

private:
   Internal::RNTupleComposerEntry *fProcessorEntry;
   Internal::RNTupleComposerEntry::FieldIndex_t fFieldIndex;

   RNTupleProcessorOptionalPtr(Internal::RNTupleComposerEntry *processorEntry,
                               Internal::RNTupleComposerEntry::FieldIndex_t fieldIdx)
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
      if (fProcessorEntry->IsValidField(fFieldIndex)) {
         const auto &value = fProcessorEntry->GetValue(fFieldIndex);
         return value.template GetPtr<T>();
      }

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
   friend class RNTupleComposer;

private:
   Internal::RNTupleComposerEntry *fProcessorEntry;
   Internal::RNTupleComposerEntry::FieldIndex_t fFieldIndex;

   RNTupleProcessorOptionalPtr(Internal::RNTupleComposerEntry *processorEntry,
                               Internal::RNTupleComposerEntry::FieldIndex_t fieldIdx)
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
      if (fProcessorEntry->IsValidField(fFieldIndex)) {
         const auto &value = fProcessorEntry->GetValue(fFieldIndex);
         return value.template GetPtr<void>();
      }

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
\class ROOT::Experimental::RNTupleComposer
\ingroup NTuple
\brief Interface for iterating over entries of vertically ("chained") and/or horizontally ("joined") combined RNTuples.

Example usage (see ntpl012_processor_chain.C and ntpl015_processor_join.C for bigger examples):

~~~{.cpp}
#include <ROOT/RNTupleComposer.hxx>
using ROOT::Experimental::RNTupleComposer;
using ROOT::Experimental::RNTupleOpenSpec;

std::vector<RNTupleOpenSpec> ntuples = {{"ntuple1", "ntuple1.root"}, {"ntuple2", "ntuple2.root"}};
auto processor = RNTupleComposer::CreateChain(ntuples);

auto pt = processor->RequestField<float>("pt");

for (const auto idx : *processor) {
   std::cout << "event = " << idx << ", pt = " << *pt << std::endl;
}
~~~

An RNTupleComposer is created either:
1. By providing one or more RNTupleOpenSpecs, each of which contains the name and storage location of a single RNTuple;
2. By providing a previously created RNTupleComposer.

The RNTupleComposer provides an iterator which gives access to the index of the current *global* entry of the
processor, i.e. taking into account previously processed RNTuples.

Because the schemas of each RNTuple that are part of an RNTupleComposer may not necessarily be identical, or because
it can occur that entries are only partially complete in a join-based processor, field values may be marked as
"invalid", at which point their data should not be read. This is handled by the RNTupleProcessorOptionalPtr
that is returned by RequestField().
*/
// clang-format on
class RNTupleComposer {
   friend struct ROOT::Experimental::Internal::RNTupleComposerEntryLoader; // for unit tests
   friend class RNTupleSingleComposer;
   friend class RNTupleChainComposer;
   friend class RNTupleJoinComposer;
   friend class RNTupleProcessor;

protected:
   std::string fProcessorName;
   std::shared_ptr<Internal::RNTupleComposerEntry> fEntry = nullptr;
   std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> fFieldIdxs;

   /// Total number of entries. Only to be used internally by the processor, not meant to be exposed in the public
   /// interface.
   ROOT::NTupleSize_t fNEntries = kInvalidNTupleIndex;

   ROOT::NTupleSize_t fCurrentEntryNumber = ROOT::kInvalidDescriptorId; //< Current entry number
   std::size_t fCurrentProcessorNumber = 0; //< Number of the currently connected processor in the composition chain

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Initialize the processor by creating an (initially empty) `fEntry`, or setting an existing one.
   virtual void Initialize(std::shared_ptr<Internal::RNTupleComposerEntry> entry) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the processor already has been initialized.
   bool IsInitialized() const { return fEntry != nullptr; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect fields to the page source of the processor's underlying RNTuple(s).
   ///
   /// \param[in] fieldIdxs Indices of the fields to connect.
   /// \param[in] provenance Provenance of the processor.
   /// \param[in] updateFields Whether the fields in the entry need to be updated, because the current underlying
   /// RNTuple source changed.
   virtual void Connect(const std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> &fieldIdxs,
                        const Internal::RNTupleCompositionProvenance &provenance, bool updateFields) = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number.
   ///
   /// \param[in] entryNumber Entry number to load
   ///
   /// \return `entryNumber` if the entry was successfully loaded, `kInvalidNTupleIndex` otherwise.
   virtual ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) = 0;

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
   /// \param[in] typeName Type of the field to add.
   /// \param[in] valuePtr Pointer to bind to the field's value in the entry. If this is a `nullptr`, a pointer will be
   /// created.
   /// \param[in] provenance Provenance of the processor.
   ///
   /// \return The index of the newly added field in the entry.
   ///
   /// In case the field was already present in the entry, the index of the existing field is returned.
   virtual Internal::RNTupleComposerEntry::FieldIndex_t
   AddFieldToEntry(const std::string &fieldName, const std::string &typeName, void *valuePtr,
                   const Internal::RNTupleCompositionProvenance &provenance) = 0;

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
   /// \brief Create a new base RNTupleComposer.
   ///
   /// \param[in] processorName Name of the processor. By default, this is the name of the underlying RNTuple for
   /// RNTupleSingleComposer, the name of the first processor for RNTupleChainComposer, or the name of the primary
   /// RNTuple for RNTupleJoinComposer.
   RNTupleComposer(std::string_view processorName) : fProcessorName(processorName) {}

public:
   RNTupleComposer(const RNTupleComposer &) = delete;
   RNTupleComposer(RNTupleComposer &&) = delete;
   RNTupleComposer &operator=(const RNTupleComposer &) = delete;
   RNTupleComposer &operator=(RNTupleComposer &&) = delete;
   virtual ~RNTupleComposer() = default;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the entry number that is currently loaded.
   ROOT::NTupleSize_t GetCurrentEntryNumber() const { return fCurrentEntryNumber; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the number of the inner processor currently being read.
   ///
   /// This method is only relevant for the RNTupleChainComposer. For the other processors, 0 is always returned.
   std::size_t GetCurrentProcessorNumber() const { return fCurrentProcessorNumber; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the name of the processor.
   ///
   /// Unless this name was explicitly specified during creation of the processor, this is the name of the underlying
   /// RNTuple for RNTupleSingleComposer, the name of the first processor for RNTupleChainComposer, or the name of the
   /// primary processor for RNTupleJoinComposer.
   const std::string &GetProcessorName() const { return fProcessorName; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Request access to a field for reading during processing.
   ///
   /// \tparam T Type of the requested field.
   ///
   /// \param[in] fieldName Name of the requested field.
   /// \param[in] valuePtr Pointer to bind to the field's value in the entry. If this is a `nullptr`, a pointer will be
   /// created.
   ///
   /// \return An RNTupleProcessorOptionalPtr of type `T`, which provides access to the field's value.
   ///
   /// \warning Provide a `valuePtr` with care! Values may not always be valid for every entry during processing, for
   /// example when a field is not present in one of the chained processors or when during a join operation, no matching
   /// entry in the auxiliary processor can be found. Reading `valuePtr` as-is therefore comes with the risk of reading
   /// invalid data. After passing a pointer to `RequestField`, we *strongly* recommend only accessing its data through
   /// the interface of the returned `RNTupleProcessorOptionalPtr`, to ensure that only valid data can be read.
   template <typename T>
   RNTupleProcessorOptionalPtr<T> RequestField(const std::string &fieldName, void *valuePtr = nullptr)
   {
      Initialize(fEntry);
      std::string typeName{};
      if constexpr (!std::is_void_v<T>) {
         typeName = ROOT::Internal::GetRenormalizedTypeName(typeid(T));
      }
      auto fieldIdx = AddFieldToEntry(fieldName, typeName, valuePtr, Internal::RNTupleCompositionProvenance());
      return RNTupleProcessorOptionalPtr<T>(fEntry.get(), fieldIdx);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Request access to a field for reading during processing.
   ///
   /// \param[in] fieldName Name of the requested field.
   /// \param[in] typeName Type of the requested field.
   /// \param[in] valuePtr Pointer to bind to the field's value in the entry. If this is a `nullptr`, a pointer will be
   /// created.
   ///
   /// \return An void-type RNTupleProcessorOptionalPtr, which provides access to the field's value.
   ///
   /// \warning Provide a `valuePtr` with care! Values may not always be valid for every entry during processing, for
   /// example when a field is not present in one of the chained processors or when during a join operation, no matching
   /// entry in the auxiliary processor can be found. Reading `valuePtr` as-is therefore comes with the risk of reading
   /// invalid data. After passing a pointer to `RequestField`, we *strongly* recommend only accessing its data through
   /// the interface of the returned `RNTupleProcessorOptionalPtr`, to ensure that only valid data can be read.
   RNTupleProcessorOptionalPtr<void>
   RequestField(const std::string &fieldName, const std::string &typeName, void *valuePtr = nullptr)
   {
      Initialize(fEntry);
      auto fieldIdx = AddFieldToEntry(fieldName, typeName, valuePtr, Internal::RNTupleCompositionProvenance());
      return RNTupleProcessorOptionalPtr<void>(fEntry.get(), fieldIdx);
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

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleComposer for a single RNTuple.
   ///
   /// \param[in] ntuple The name and storage location of the RNTuple to process.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the input RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleComposer.
   static std::unique_ptr<RNTupleComposer> Create(RNTupleOpenSpec ntuple, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleComposer for a *chain* (i.e., a vertical combination) of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the RNTuples to process.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleComposer.
   static std::unique_ptr<RNTupleComposer>
   CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleComposer for a *chain* (i.e., a vertical combination) of other RNTupleComposers.
   ///
   /// \param[in] innerProcessors A list with the processors to chain.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first inner processor is
   /// used.
   ///
   /// \return A pointer to the newly created RNTupleComposer.
   static std::unique_ptr<RNTupleComposer>
   CreateChain(std::vector<std::unique_ptr<RNTupleComposer>> innerProcessors, std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleComposer for a *join* (i.e., a horizontal combination) of RNTuples.
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
   /// \return A pointer to the newly created RNTupleComposer.
   static std::unique_ptr<RNTupleComposer> CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                      const std::vector<std::string> &joinFields,
                                                      std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleComposer for a *join* (i.e., a horizontal combination) of RNTuples.
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
   /// \return A pointer to the newly created RNTupleComposer.
   static std::unique_ptr<RNTupleComposer>
   CreateJoin(std::unique_ptr<RNTupleComposer> primaryProcessor, std::unique_ptr<RNTupleComposer> auxProcessor,
              const std::vector<std::string> &joinFields, std::string_view processorName = "");
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleSingleComposer
\ingroup NTuple
\brief Composer specialization for processing a single RNTuple.
*/
// clang-format on
class RNTupleSingleComposer : public RNTupleComposer {
   friend class RNTupleComposer;

private:
   RNTupleOpenSpec fNTupleSpec;
   std::unique_ptr<ROOT::Internal::RPageSource> fPageSource;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create a new field and connect it to the processor's page source.
   ///
   /// \param[in] qualifiedFieldName Name of the field to add, prefixed with its parent fields, if applicable.
   /// \param[in] typeName Type of the field to add.
   ///
   /// \return The newly created field.
   /// \throws ROOT::RException In case the requested field cannot be found on disk.
   std::unique_ptr<ROOT::RFieldBase>
   CreateAndConnectField(const std::string &qualifiedFieldName, const std::string &typeName);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Initialize the processor by creating an (initially empty) `fEntry`, or setting an existing one.
   ///
   /// At this point, the page source for the underlying RNTuple of the processor will be created and opened.
   void Initialize(std::shared_ptr<Internal::RNTupleComposerEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   void Connect(const std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::LoadEntry
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
   /// \sa RNTupleComposer::CanReadFieldFromDisk()
   bool CanReadFieldFromDisk(std::string_view fieldName) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   /// \sa RNTupleComposer::AddFieldToEntry()
   Internal::RNTupleComposerEntry::FieldIndex_t AddFieldToEntry(
      const std::string &fieldName, const std::string &typeName, void *valuePtr = nullptr,
      const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Processor-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::PrintStructureImpl
   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleComposer for processing a single RNTuple.
   ///
   /// \param[in] ntuple The source specification (name and storage location) for the RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleComposer::Create, this is
   /// the name of the underlying RNTuple.
   RNTupleSingleComposer(RNTupleOpenSpec ntuple, std::string_view processorName);

public:
   RNTupleSingleComposer(const RNTupleSingleComposer &) = delete;
   RNTupleSingleComposer(RNTupleSingleComposer &&) = delete;
   RNTupleSingleComposer &operator=(const RNTupleSingleComposer &) = delete;
   RNTupleSingleComposer &operator=(RNTupleSingleComposer &&) = delete;
   ~RNTupleSingleComposer() override
   {
      // The entry's fields need to be deleted before fPageSource.
      if (fEntry)
         fEntry->Clear();
   };
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleChainComposer
\ingroup NTuple
\brief Composer specialization for vertically combined (*chained*) RNTupleComposers.
*/
// clang-format on
class RNTupleChainComposer : public RNTupleComposer {
   friend class RNTupleComposer;

private:
   std::vector<std::unique_ptr<RNTupleComposer>> fInnerProcessors;
   std::vector<ROOT::NTupleSize_t> fInnerNEntries;

   Internal::RNTupleCompositionProvenance fProvenance;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Initialize the processor by creating an (initially empty) `fEntry`, or setting an existing one.
   void Initialize(std::shared_ptr<Internal::RNTupleComposerEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   ///
   /// \sa RNTupleComposer::Connect()
   void Connect(const std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Update the entry to reflect any missing fields in the current inner processor.
   void ConnectInnerProcessor(std::size_t composerNumber);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ///
   /// \note This requires opening all underlying RNTuples being processed in the chain, and could become costly!
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \sa RNTupleComposer::CanReadFieldFromDisk()
   bool CanReadFieldFromDisk(std::string_view fieldName) final
   {
      return fInnerProcessors[fCurrentProcessorNumber]->CanReadFieldFromDisk(fieldName);
   }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add a field to the entry.
   ///
   /// \sa RNTupleComposer::AddFieldToEntry()
   Internal::RNTupleComposerEntry::FieldIndex_t AddFieldToEntry(
      const std::string &fieldName, const std::string &typeName, void *valuePtr = nullptr,
      const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Composer-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::PrintStructureImpl
   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleChainComposer.
   ///
   /// \param[in] ntuples The source specification (name and storage location) for each RNTuple to process.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleComposer::CreateChain, this
   /// is the name of the first inner processor.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainComposer(std::vector<std::unique_ptr<RNTupleComposer>> processors, std::string_view processorName);

public:
   RNTupleChainComposer(const RNTupleChainComposer &) = delete;
   RNTupleChainComposer(RNTupleChainComposer &&) = delete;
   RNTupleChainComposer &operator=(const RNTupleChainComposer &) = delete;
   RNTupleChainComposer &operator=(RNTupleChainComposer &&) = delete;
   ~RNTupleChainComposer() override = default;
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleJoinComposer
\ingroup NTuple
\brief Composer specialization for horizontally combined (*joined*) RNTupleComposers.
*/
// clang-format on
class RNTupleJoinComposer : public RNTupleComposer {
   friend class RNTupleComposer;

private:
   std::unique_ptr<RNTupleComposer> fPrimaryProcessor;
   std::unique_ptr<RNTupleComposer> fAuxiliaryProcessor;

   std::vector<std::string> fJoinFieldNames;
   std::set<Internal::RNTupleComposerEntry::FieldIndex_t> fJoinFieldIdxs;

   std::unique_ptr<Internal::RNTupleJoinTable> fJoinTable;
   bool fJoinTableIsBuilt = false;

   std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> fAuxiliaryFieldIdxs;

   /// \brief Initialize the processor by creating an (initially empty) `fEntry`, or setting an existing one.
   void Initialize(std::shared_ptr<Internal::RNTupleComposerEntry> entry = nullptr) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Connect the provided fields indices in the entry to their on-disk fields.
   ///
   /// \sa RNTupleComposer::Connect()
   void Connect(const std::unordered_set<Internal::RNTupleComposerEntry::FieldIndex_t> &fieldIdxs,
                const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance(),
                bool updateFields = false) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number of the primary processor.
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set the validity for all fields in the auxiliary processor at once.
   void SetAuxiliaryFieldValidity(bool validity);

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Check if a field exists on-disk and can be read by the processor.
   ///
   /// \sa RNTupleComposer::CanReadFieldFromDisk()
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
   /// \sa RNTupleComposer::AddFieldToEntry()
   Internal::RNTupleComposerEntry::FieldIndex_t AddFieldToEntry(
      const std::string &fieldName, const std::string &typeName, void *valuePtr = nullptr,
      const Internal::RNTupleCompositionProvenance &provenance = Internal::RNTupleCompositionProvenance()) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Composer-specific implementation for printing its structure, called by PrintStructure().
   ///
   /// \sa ROOT::Experimental::RNTupleComposer::PrintStructureImpl
   void PrintStructureImpl(std::ostream &output) const final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Construct a new RNTupleJoinComposer.
   /// \param[in] primaryProcessor The primary processor. Its entries are processed in sequential order.
   /// \param[in] auxProcessor The processor to join the primary processor with. The order in which its entries are
   /// processed is determined by the primary processor and doesn't necessarily have to be sequential.
   /// \param[in] joinFields The names of the fields on which to join, in case the specified processors are unaligned.
   /// The join is made based on the combined join field values, and therefore each field has to be present in each
   /// specified processor. If an empty list is provided, it is assumed that the processors are fully aligned.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleComposer::CreateJoin, this
   /// is the name of the primary processor.
   RNTupleJoinComposer(std::unique_ptr<RNTupleComposer> primaryProcessor, std::unique_ptr<RNTupleComposer> auxProcessor,
                       const std::vector<std::string> &joinFields, std::string_view processorName);

public:
   RNTupleJoinComposer(const RNTupleJoinComposer &) = delete;
   RNTupleJoinComposer operator=(const RNTupleJoinComposer &) = delete;
   RNTupleJoinComposer(RNTupleJoinComposer &&) = delete;
   RNTupleJoinComposer operator=(RNTupleJoinComposer &&) = delete;
   ~RNTupleJoinComposer() override = default;
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleComposer
