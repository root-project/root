// Author: Enrico Guiraud CERN 09/2020
// Author: Vincenzo Eduardo Padulano CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RTREECOLUMNREADER
#define ROOT_RDF_RTREECOLUMNREADER

#include "RColumnReaderBase.hxx"
#include <ROOT/RVec.hxx>
#include "ROOT/RDF/Utils.hxx"
#include <Rtypes.h> // Long64_t, R__CLING_PTRCHECK

#include <array>
#include <memory>
#include <string>
#include <cstddef>
#include <variant>

class TBranch;
class TBuffer;

class TTreeReader;
template <typename T>
class TTreeReaderValue;

namespace ROOT {
namespace Internal {

class TTreeReaderOpaqueValue;
class TTreeReaderUntypedArray;
class TTreeReaderUntypedValue;

namespace RDF {

class R__CLING_PTRCHECK(off) RTreeOpaqueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   TTreeReader &fTreeReader;
   std::unique_ptr<ROOT::Internal::TTreeReaderOpaqueValue> fTreeValue;
   std::uint64_t fLastEntry;
   ROOT::RVec<bool> fIsCachedValueValidAtIdx;

   void *GetImpl(std::size_t) override;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) override;

public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeOpaqueColumnReader(TTreeReader &r, std::string_view colName, std::size_t maxBulkSize);

   // Rule of five

   RTreeOpaqueColumnReader(const RTreeOpaqueColumnReader &) = delete;
   RTreeOpaqueColumnReader &operator=(const RTreeOpaqueColumnReader &) = delete;
   RTreeOpaqueColumnReader(RTreeOpaqueColumnReader &&) = delete;
   RTreeOpaqueColumnReader &operator=(RTreeOpaqueColumnReader &&) = delete;
   ~RTreeOpaqueColumnReader() final; // Define destructor when data member type is fully defined
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderUntypedValue
class R__CLING_PTRCHECK(off) RTreeUntypedValueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   friend class RTreeGenericValueColumnReader;

   TTreeReader &fTreeReader;
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedValue> fTreeValue;
   ROOT::RVec<std::byte> fCachedValues;
   ROOT::RVec<void *> fCachedBranchAddresses;
   ROOT::RVec<void *> fCachedBranchAddressesOfAddresses;
   std::function<void(void *)> fObjDeleter;
   ROOT::RVec<std::unique_ptr<void, std::function<void(void *)>>> fCachedBranchAddressesRAII;
   ROOT::RVec<bool> fIsCachedValueValidAtIdx;
   std::size_t fValueSize;
   std::uint64_t fLastEntry;
   void *fAddrOfTypeThatCannotBeReadInBulk;
   std::string fColName;

   void *GetImpl(std::size_t) override;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) override;

public:
   RTreeUntypedValueColumnReader(TTreeReader &r, std::string_view colName, std::string_view typeName,
                                 std::size_t maxBulkSize);

   // Rule of five

   RTreeUntypedValueColumnReader(const RTreeUntypedValueColumnReader &) = delete;
   RTreeUntypedValueColumnReader &operator=(const RTreeUntypedValueColumnReader &) = delete;
   RTreeUntypedValueColumnReader(RTreeUntypedValueColumnReader &&) = delete;
   RTreeUntypedValueColumnReader &operator=(RTreeUntypedValueColumnReader &&) = delete;
   ~RTreeUntypedValueColumnReader() final; // Define destructor when data member type is fully defined
};

class R__CLING_PTRCHECK(off) RTreeScalarBulkColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
public:
   using BulkLoadFunc_t = void (*)(ROOT::RVec<std::byte> &, TBranch &, std::size_t, std::uint64_t, std::uint64_t,
                                   std::unique_ptr<TBuffer> &, std::size_t &, std::uint64_t &, Long64_t &,
                                   std::uint64_t &);

private:
   friend class RTreeLeafCountArrayBulkColumnReader;
   friend class RTreeGenericValueColumnReader;

   TTreeReader &fTreeReader;

   // Used to keep track of the current active branch
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedValue> fTreeValue;

   std::size_t fBulkSize;
   std::size_t fValueSize;
   ROOT::RVec<std::byte> fCachedValues;
   bool fIsBulkValid;

   std::uint64_t fLastEntry;

   const std::type_info &fUserRequestedTypeInfo;

   std::unique_ptr<TBuffer> fCachedBuffer;
   std::size_t fBranchBulkNumberOfEntries;
   std::uint64_t fEntriesReadFromCachedBuffer;
   Long64_t fLastBasketBegin;
   std::uint64_t fLastChainOffset;

   void *GetImpl(std::size_t) final;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) final;

public:
   RTreeScalarBulkColumnReader(TTreeReader &r, std::string_view colName, std::string_view typeName,
                               const std::type_info &ti, std::size_t maxBulkSize);

   RTreeScalarBulkColumnReader(const RTreeScalarBulkColumnReader &) = delete;
   RTreeScalarBulkColumnReader &operator=(const RTreeScalarBulkColumnReader &) = delete;
   RTreeScalarBulkColumnReader(RTreeScalarBulkColumnReader &&) = delete;
   RTreeScalarBulkColumnReader &operator=(RTreeScalarBulkColumnReader &&) = delete;
   ~RTreeScalarBulkColumnReader() final; // Define destructor when data member type is fully defined
};

class R__CLING_PTRCHECK(off) RTreeGenericValueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
private:
   std::variant<std::unique_ptr<RTreeUntypedValueColumnReader>, std::unique_ptr<RTreeScalarBulkColumnReader>> fReader;

   void *GetImpl(std::size_t) final;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) final;

public:
   RTreeGenericValueColumnReader(TTreeReader &r, std::string_view colName, std::string_view typeName,
                                 const std::type_info &ti, std::size_t maxBulkSize);

   RTreeGenericValueColumnReader(const RTreeGenericValueColumnReader &) = delete;
   RTreeGenericValueColumnReader &operator=(const RTreeGenericValueColumnReader &) = delete;
   RTreeGenericValueColumnReader(RTreeGenericValueColumnReader &&) = delete;
   RTreeGenericValueColumnReader &operator=(RTreeGenericValueColumnReader &&) = delete;
   ~RTreeGenericValueColumnReader() final; // Define destructor when data member type is fully defined
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderUntypedValue
class R__CLING_PTRCHECK(off) RTreeUntypedArrayColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
public:
   enum class ECollectionType {
      kRVec,
      kStdArray,
      kRVecBool,
      kStdVector
   };

   RTreeUntypedArrayColumnReader(TTreeReader &r, std::string_view colName, std::string_view valueTypeName,
                                 ECollectionType collType, std::size_t maxBulkSize);

   // Rule of five

   RTreeUntypedArrayColumnReader(const RTreeUntypedArrayColumnReader &) = delete;
   RTreeUntypedArrayColumnReader &operator=(const RTreeUntypedArrayColumnReader &) = delete;
   RTreeUntypedArrayColumnReader(RTreeUntypedArrayColumnReader &&) = delete;
   RTreeUntypedArrayColumnReader &operator=(RTreeUntypedArrayColumnReader &&) = delete;
   ~RTreeUntypedArrayColumnReader() final; // Define destructor when data member type is fully defined

private:
   friend class RTreeGenericCollectionColumnReader;

   TTreeReader &fTreeReader;
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedArray> fTreeArray;
   ECollectionType fCollectionType;

   std::uint64_t fLastEntry;

   /// The size of the collection value type.
   std::size_t fValueSize;

   // We absolutely require stable addresses for all collections in the bulk, so we need to instantiate a bulk of
   // collections at construction time and then always reuse those collections.
   using CacheStorage_t = std::variant<std::vector<ROOT::RVec<std::byte>>, std::vector<std::vector<std::byte>>>;
   CacheStorage_t fCachedValues;
   ROOT::RVec<bool> fIsCachedValueValidAtIdx;

   void *GetImpl(std::size_t) override;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) override;

   void LoadStdArray(std::size_t entryInBulk);
   void LoadStdVector(std::size_t entryInBulk);
   void LoadRVec(std::size_t entryInBulk);
};

class R__CLING_PTRCHECK(off) RMaskedColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
   TTreeReader &fTreeReader;
   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> fValueReader;
   std::unique_ptr<TTreeReaderValue<uint64_t>> fTreeValueMask;
   unsigned int fMaskIndex = 0;

   std::uint64_t fLastEntry;
   ROOT::RVec<bool> fIsCachedValueValidAtIdx;

   void *GetImpl(std::size_t) override;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) override;

public:
   RMaskedColumnReader(TTreeReader &r, std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> valueReader,
                       std::string_view maskName, unsigned int maskIndex, std::size_t maxBulkSize);
   ~RMaskedColumnReader() override;
   void SetColumnReader(std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> reader)
   {
      fValueReader = std::move(reader);
   }
};

class R__CLING_PTRCHECK(off) RTreeLeafCountArrayBulkColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
public:
   using BulkLoadFunc_t = void (*)(ROOT::RVec<std::byte> &, ROOT::RVec<ROOT::RVec<std::byte>> &, TBranch &, std::size_t,
                                   std::uint64_t, std::uint64_t, ROOT::RVec<std::byte> &, std::uint64_t &,
                                   std::unique_ptr<TBuffer> &, std::size_t &, std::uint64_t &, Long64_t &,
                                   std::uint64_t &);

private:
   friend class RTreeGenericCollectionColumnReader;

   TTreeReader &fTreeReader;
   // Used to keep track of the current active branch
   std::unique_ptr<ROOT::Internal::TTreeReaderUntypedArray> fTreeArray;

   ROOT::RVec<std::byte> fCachedValues;
   ROOT::RVec<ROOT::RVec<std::byte>> fCachedViews;

   bool fIsBulkValid;

   std::uint64_t fLastEntry;

   const std::type_info &fUserRequestedValueTypeInfo;

   std::unique_ptr<RTreeScalarBulkColumnReader> fLeafCountSizesReader;
   const std::type_info &fLeafCountTypeInfo;

   std::uint64_t fBufElementsOffset;

   std::unique_ptr<TBuffer> fCachedBuffer;
   std::size_t fBranchBulkNumberOfEntries;
   std::uint64_t fEntriesReadFromCachedBuffer;
   Long64_t fLastBasketBegin;
   std::uint64_t fLastChainOffset;

   void *GetImpl(std::size_t) final;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) final;

public:
   RTreeLeafCountArrayBulkColumnReader(TTreeReader &r, std::string_view arrayColName, std::string_view valueTypeName,
                                       const std::type_info &valueTypeInfo, const std::type_info &leafCountTypeInfo,
                                       std::unique_ptr<RTreeScalarBulkColumnReader> sizeReader,
                                       std::size_t maxBulkSize);

   RTreeLeafCountArrayBulkColumnReader(const RTreeLeafCountArrayBulkColumnReader &) = delete;
   RTreeLeafCountArrayBulkColumnReader &operator=(const RTreeLeafCountArrayBulkColumnReader &) = delete;
   RTreeLeafCountArrayBulkColumnReader(RTreeLeafCountArrayBulkColumnReader &&) = delete;
   RTreeLeafCountArrayBulkColumnReader &operator=(RTreeLeafCountArrayBulkColumnReader &&) = delete;
   ~RTreeLeafCountArrayBulkColumnReader() final; // Define destructor when data member type is fully defined
};

class R__CLING_PTRCHECK(off) RTreeGenericCollectionColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
private:
   std::variant<std::unique_ptr<RTreeUntypedArrayColumnReader>, std::unique_ptr<RTreeLeafCountArrayBulkColumnReader>>
      fReader;

   void *GetImpl(std::size_t) final;
   void LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &) final;

public:
   RTreeGenericCollectionColumnReader(
      TTreeReader &r, std::string_view arrayColName, std::string_view valueTypeName,
      const std::type_info &valueTypeInfo,
      ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType collectionType, std::size_t maxBulkSize);

   RTreeGenericCollectionColumnReader(const RTreeGenericCollectionColumnReader &) = delete;
   RTreeGenericCollectionColumnReader &operator=(const RTreeGenericCollectionColumnReader &) = delete;
   RTreeGenericCollectionColumnReader(RTreeGenericCollectionColumnReader &&) = delete;
   RTreeGenericCollectionColumnReader &operator=(RTreeGenericCollectionColumnReader &&) = delete;
   ~RTreeGenericCollectionColumnReader() final; // Define destructor when data member type is fully defined
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
