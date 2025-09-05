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
#include <ROOT/RFieldToken.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleJoinTable.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
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
   std::string fProcessorName;
   std::unique_ptr<ROOT::REntry> fEntry;
   std::unique_ptr<ROOT::RNTupleModel> fModel;

   /// Total number of entries. Only to be used internally by the processor, not meant to be exposed in the public
   /// interface.
   ROOT::NTupleSize_t fNEntries = kInvalidNTupleIndex;

   ROOT::NTupleSize_t fNEntriesProcessed = 0;  //< Total number of entries processed so far
   ROOT::NTupleSize_t fCurrentEntryNumber = 0; //< Current processor entry number
   std::size_t fCurrentProcessorNumber = 0;    //< Number of the currently open inner processor

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
   virtual void SetEntryPointers(const ROOT::REntry &entry, std::string_view fieldNamePrefix = "") = 0;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor
   virtual ROOT::NTupleSize_t GetNEntries() = 0;

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
   /// \param[in] model The RNTupleModel representing the entries returned by the processor.
   ///
   /// \note Before processing, a model *must* exist. However, this is handled downstream by the RNTupleProcessor's
   /// factory functions (CreateSingle, CreateChain and CreateJoin) and constructors.
   RNTupleProcessor(std::string_view processorName, std::unique_ptr<ROOT::RNTupleModel> model)
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
   /// primary processor for RNTupleJoinProcessor.
   const std::string &GetProcessorName() const { return fProcessorName; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the model used by the processor.
   const ROOT::RNTupleModel &GetModel() const { return *fModel; }

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get a reference to the entry used by the processor.
   ///
   /// \return A reference to the entry used by the processor.
   const ROOT::REntry &GetEntry() const { return *fEntry; }

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
      using iterator_category = std::forward_iterator_tag;
      using iterator = RIterator;
      using value_type = ROOT::REntry;
      using difference_type = std::ptrdiff_t;
      using pointer = ROOT::REntry *;
      using reference = const ROOT::REntry &;

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
   /// \param[in] processorName The name to give to the processor. If empty, the name of the input RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> Create(RNTupleOpenSpec ntuple,
                                                   std::unique_ptr<ROOT::RNTupleModel> model = nullptr,
                                                   std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of RNTuples.
   ///
   /// \param[in] ntuples A list specifying the names and locations of the RNTuples to process.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the descriptor of the first RNTuple specified.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateChain(std::vector<RNTupleOpenSpec> ntuples,
                                                        std::unique_ptr<ROOT::RNTupleModel> model = nullptr,
                                                        std::string_view processorName = "");

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Create an RNTupleProcessor for a *chain* (i.e., a vertical combination) of other RNTupleProcessors.
   ///
   /// \param[in] innerProcessors A list with the processors to chain.
   /// \param[in] model An RNTupleModel specifying which fields can be read by the processor. If no model is provided,
   /// one will be created based on the model used by the first inner processor.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the first inner processor is
   /// used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor> CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                        std::unique_ptr<ROOT::RNTupleModel> model = nullptr,
                                                        std::string_view processorName = "");

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
   /// \param[in] primaryModel An RNTupleModel specifying which fields from the primary RNTuple can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the primary RNTuple.
   /// \param[in] auxModel An RNTupleModel specifying which fields from the auxiliary RNTuple can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the auxiliary RNTuple.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the primary RNTuple is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple, const std::vector<std::string> &joinFields,
              std::unique_ptr<ROOT::RNTupleModel> primaryModel = nullptr,
              std::unique_ptr<ROOT::RNTupleModel> auxModel = nullptr, std::string_view processorName = "");

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
   /// \param[in] primaryModel An RNTupleModel specifying which fields from the primary processor can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the primary processor.
   /// \param[in] auxModel An RNTupleModel specifying which fields from the auxiliary processor can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the auxiliary processor.
   /// \param[in] processorName The name to give to the processor. If empty, the name of the primary processor is used.
   ///
   /// \return A pointer to the newly created RNTupleProcessor.
   static std::unique_ptr<RNTupleProcessor>
   CreateJoin(std::unique_ptr<RNTupleProcessor> primaryProcessor, std::unique_ptr<RNTupleProcessor> auxProcessor,
              const std::vector<std::string> &joinFields, std::unique_ptr<ROOT::RNTupleModel> primaryModel = nullptr,
              std::unique_ptr<ROOT::RNTupleModel> auxModel = nullptr, std::string_view processorName = "");
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
   void Connect();

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided (global) entry number (i.e., considering all RNTuples in this
   /// processor).
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \sa ROOT::Experimental::RNTupleProcessor::SetEntryPointers.
   void SetEntryPointers(const ROOT::REntry &entry, std::string_view fieldNamePrefix) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final
   {
      Connect();
      return fNEntries;
   }

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
   /// \param[in] model The model that specifies which fields should be read by the processor.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::Create, this is
   /// the name of the underlying RNTuple.
   RNTupleSingleProcessor(RNTupleOpenSpec ntuple, std::unique_ptr<ROOT::RNTupleModel> model,
                          std::string_view processorName);

public:
   RNTupleSingleProcessor(const RNTupleSingleProcessor &) = delete;
   RNTupleSingleProcessor(RNTupleSingleProcessor &&) = delete;
   RNTupleSingleProcessor &operator=(const RNTupleSingleProcessor &) = delete;
   RNTupleSingleProcessor &operator=(RNTupleSingleProcessor &&) = delete;
   ~RNTupleSingleProcessor() override { fModel.release(); };
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
   void SetEntryPointers(const ROOT::REntry &, std::string_view fieldNamePrefix) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ///
   /// \note This requires opening all underlying RNTuples being processed in the chain, and could become costly!
   ROOT::NTupleSize_t GetNEntries() final;

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
   /// \param[in] model The model that specifies which fields should be read by the processor. The pointer returned by
   /// RNTupleModel::MakeField can be used to access a field's value during the processor iteration. When no model is
   /// specified, it is created from the descriptor of the first RNTuple specified in `ntuples`.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateChain, this
   /// is the name of the first inner processor.
   ///
   /// RNTuples are processed in the order in which they are specified.
   RNTupleChainProcessor(std::vector<std::unique_ptr<RNTupleProcessor>> processors,
                         std::unique_ptr<ROOT::RNTupleModel> model, std::string_view processorName);

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

   /// Tokens representing the join fields present in the primary processor.
   std::vector<ROOT::RFieldToken> fJoinFieldTokens;
   std::unique_ptr<Internal::RNTupleJoinTable> fJoinTable;
   bool fJoinTableIsBuilt = false;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Load the entry identified by the provided entry number of the primary processor.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::LoadEntry
   ROOT::NTupleSize_t LoadEntry(ROOT::NTupleSize_t entryNumber) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \sa ROOT::Experimental::RNTupleProcessor::SetEntryPointers.
   void SetEntryPointers(const ROOT::REntry &, std::string_view fieldNamePrefix) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Get the total number of entries in this processor.
   ROOT::NTupleSize_t GetNEntries() final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Add the entry mappings for this processor to the provided join table.
   ///
   /// \sa ROOT::Experimental::RNTupleProcessor::AddEntriesToJoinTable
   void AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset = 0) final;

   /////////////////////////////////////////////////////////////////////////////
   /// \brief Set fModel by combining the primary and auxiliary models.
   ///
   /// \param[in] primaryModel The model of the primary processor.
   /// \param[in] auxModel Model of the auxiliary processors.
   ///
   /// To prevent field name clashes when one or more models have fields with duplicate names, fields from each
   /// auxiliary model are stored as a anonymous record, and subsequently registered as subfields in the join model.
   /// This way, they can be accessed from the processor's entry as `auxNTupleName.fieldName`.
   void SetModel(std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::unique_ptr<ROOT::RNTupleModel> auxModel);

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
   /// \param[in] primaryModel An RNTupleModel specifying which fields from the primary processor can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the primary processor.
   /// \param[in] auxModel An RNTupleModel specifying which fields from the auxiliary processor can be read by the
   /// processor. If no model is provided, one will be created based on the descriptor of the auxiliary processor.
   /// \param[in] processorName Name of the processor. Unless specified otherwise in RNTupleProcessor::CreateJoin, this
   /// is the name of the primary processor.
   RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                        std::unique_ptr<RNTupleProcessor> auxProcessor, const std::vector<std::string> &joinFields,
                        std::unique_ptr<ROOT::RNTupleModel> primaryModel, std::unique_ptr<ROOT::RNTupleModel> auxModel,
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
