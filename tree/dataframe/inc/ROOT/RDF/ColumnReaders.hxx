// Author: Enrico Guiraud CERN 08/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_COLUMNREADERS
#define ROOT_RDF_COLUMNREADERS

#include <ROOT/RDF/RCustomColumnBase.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/TypeTraits.hxx>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

using namespace ROOT::TypeTraits;
namespace RDFDetail = ROOT::Detail::RDF;

void CheckCustomColumn(RDFDetail::RCustomColumnBase &customColumn, const std::type_info &tid);

/**
\class ROOT::Internal::RDF::RColumnReaderBase
\ingroup dataframe
\brief Pure virtual base class for all column reader types
\tparam T The type of the column

This pure virtual class provides a common base class for the different column reader types, e.g. RTreeColumnReader and
RDSColumnReader.
**/
template <typename T>
class RColumnReaderBase {
public:
   virtual ~RColumnReaderBase() = default;
   /// Return the column value for the given entry. Called at most once per entry.
   virtual T &Get(Long64_t entry) = 0;
   /// Perform clean-up operations if needed. Called at the end of a processing task.
   virtual void Reset() {}
};

/// Column reader for defined (aka custom) columns.
template <typename T>
class R__CLING_PTRCHECK(off) RDefineReader final : public RColumnReaderBase<T> {
   /// Non-owning reference to the node responsible for the custom column. Needed when querying custom values.
   RDFDetail::RCustomColumnBase &fCustomColumn;

   /// Non-owning ptr to the value of a custom column.
   T *fCustomValuePtr = nullptr;

   /// The slot this value belongs to.
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();
public:
   RDefineReader(unsigned int slot, RDFDetail::RCustomColumnBase &customColumn)
      : fCustomColumn(customColumn), fCustomValuePtr(static_cast<T *>(customColumn.GetValuePtr(slot))), fSlot(slot)
   {
      CheckCustomColumn(customColumn, typeid(T));
   }

   T &Get(Long64_t entry) final
   {
      fCustomColumn.Update(fSlot, entry);
      return *fCustomValuePtr;
   }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderValues
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader final : public RColumnReaderBase<T> {
   std::unique_ptr<TTreeReaderValue<T>> fTreeValue;

public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeValue(std::make_unique<TTreeReaderValue<T>>(r, colName.c_str()))
   {
   }

   T &Get(Long64_t) final { return **fTreeValue; }

   /// Delete the TTreeReaderValue object.
   //
   // Without this call, a race condition is present in which a TTreeReader
   // and its TTreeReader{Value,Array}s can be deleted concurrently:
   // - Thread #1) a task ends and pushes back processing slot
   // - Thread #2) a task starts and overwrites thread-local TTreeReaderValues
   // - Thread #1) first task deletes TTreeReader
   // See https://github.com/root-project/root/commit/26e8ace6e47de6794ac9ec770c3bbff9b7f2e945
   void Reset() final { fTreeValue.reset(); }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderArrays.
///
/// TTreeReaderArrays are used whenever the RDF column type is RVec<T>.
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<T>> final : public RColumnReaderBase<RVec<T>> {
   std::unique_ptr<TTreeReaderArray<T>> fTreeArray;

   /// Enumerator for the memory layout of the branch
   enum class EStorageType : char { kContiguous, kUnknown, kSparse };

   /// Signal whether we ever checked that the branch we are reading with a TTreeReaderArray stores array elements
   /// in contiguous memory.
   EStorageType fStorageType = EStorageType::kUnknown;

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout.
   RVec<T> fRVec;

   /// Whether we already printed a warning about performing a copy of the TTreeReaderArray contents
   bool fCopyWarningPrinted = false;

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderArray<T>>(r, colName.c_str()))
   {
   }

   RVec<T> &Get(Long64_t) final
   {
      auto &readerArray = *fTreeArray;
      // We only use TTreeReaderArrays to read columns that users flagged as type `RVec`, so we need to check
      // that the branch stores the array as contiguous memory that we can actually wrap in an `RVec`.
      // Currently we need the first entry to have been loaded to perform the check
      // TODO Move check to constructor once ROOT-10823 is fixed and TTreeReaderArray itself exposes this information
      const auto arrSize = readerArray.GetSize();
      if (EStorageType::kUnknown == fStorageType && arrSize > 1) {
         // We can decide since the array is long enough
         fStorageType = EStorageType::kContiguous;
         for (auto i = 0u; i < arrSize - 1; ++i) {
            if ((char *)&readerArray[i + 1] - (char *)&readerArray[i] != sizeof(T)) {
               fStorageType = EStorageType::kSparse;
               break;
            }
         }
      }

      const auto readerArraySize = readerArray.GetSize();
      if (EStorageType::kContiguous == fStorageType ||
          (EStorageType::kUnknown == fStorageType && readerArray.GetSize() < 2)) {
         if (readerArraySize > 0) {
            // trigger loading of the contents of the TTreeReaderArray
            // the address of the first element in the reader array is not necessarily equal to
            // the address returned by the GetAddress method
            auto readerArrayAddr = &readerArray.At(0);
            RVec<T> rvec(readerArrayAddr, readerArraySize);
            std::swap(fRVec, rvec);
         } else {
            RVec<T> emptyVec{};
            std::swap(fRVec, emptyVec);
         }
      } else {
         // The storage is not contiguous or we don't know yet: we cannot but copy into the rvec
#ifndef NDEBUG
         if (!fCopyWarningPrinted) {
            Warning("RColumnValue::Get",
                    "Branch %s hangs from a non-split branch. A copy is being performed in order "
                    "to properly read the content.",
                    readerArray.GetBranchName());
            fCopyWarningPrinted = true;
         }
#else
         (void)fCopyWarningPrinted;
#endif
         if (readerArraySize > 0) {
            RVec<T> rvec(readerArray.begin(), readerArray.end());
            std::swap(fRVec, rvec);
         } else {
            RVec<T> emptyVec{};
            std::swap(fRVec, emptyVec);
         }
      }
      return fRVec;
   }

   /// Delete the TTreeReaderArray object.
   void Reset() final { fTreeArray.reset(); }
};

/// RTreeColumnReader specialization for arrays of boolean values read via TTreeReaderArrays.
///
/// TTreeReaderArray<bool> is used whenever the RDF column type is RVec<bool>.
template <>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<bool>> final : public RColumnReaderBase<RVec<bool>> {

   std::unique_ptr<TTreeReaderArray<bool>> fTreeArray;

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout
   RVec<bool> fRVec;

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderArray<bool>>(r, colName.c_str()))
   {
   }

   // We always copy the contents of TTreeReaderArray<bool> into an RVec<bool> (never take a view into the memory
   // buffer) because the underlying memory buffer might be the one of a std::vector<bool>, which is not a contiguous
   // slab of bool values.
   // Note that this also penalizes the case in which the column type is actually bool[], but the possible performance
   // gains in this edge case is probably not worth the extra complication required to differentiate the two cases.
   RVec<bool> &Get(Long64_t) final
   {
      auto &readerArray = *fTreeArray;
      const auto readerArraySize = readerArray.GetSize();
      if (readerArraySize > 0) {
         // always perform a copy
         RVec<bool> rvec(readerArray.begin(), readerArray.end());
         std::swap(fRVec, rvec);
      } else {
         RVec<bool> emptyVec{};
         std::swap(fRVec, emptyVec);
      }
      return fRVec;
   }

   /// Delete the TTreeReaderArray object.
   void Reset() final { fTreeArray.reset(); }
};

/// Column reader type that deals with values read from RDataSources.
template <typename T>
class R__CLING_PTRCHECK(off) RDSColumnReader final : public RColumnReaderBase<T> {
   RDFDetail::RCustomColumnBase &fCustomColumn;
   T **fDSValuePtr = nullptr;
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();

public:
   RDSColumnReader(unsigned int slot, RDFDetail::RCustomColumnBase &customColumn)
      : fCustomColumn(customColumn), fDSValuePtr(static_cast<T **>(customColumn.GetValuePtr(slot))), fSlot(slot)
   {
   }

   T &Get(Long64_t entry) final
   {
      fCustomColumn.Update(fSlot, entry);
      return **fDSValuePtr;
   }
};

template <typename BranchTypeList>
struct RDFValueTuple {
};

template <typename... BranchTypes>
struct RDFValueTuple<TypeList<BranchTypes...>> {
   using type = std::tuple<std::unique_ptr<RColumnReaderBase<BranchTypes>>...>;
};

template <typename BranchTypeList>
using RDFValueTuple_t = typename RDFValueTuple<BranchTypeList>::type;

template <typename T>
std::unique_ptr<RColumnReaderBase<T>>
MakeColumnReader(unsigned int slot, RDFDetail::RCustomColumnBase *customCol, TTreeReader *r, const std::string &colName)
{
   using Ret_t = std::unique_ptr<RColumnReaderBase<T>>;

   if (customCol == nullptr)
      return Ret_t(new RTreeColumnReader<T>(*r, colName));

   if (customCol->IsDataSourceColumn())
      return Ret_t(new RDSColumnReader<T>(slot, *customCol));

   return Ret_t(new RDefineReader<T>(slot, *customCol));
}

template <typename T>
void InitColumnReadersHelper(std::unique_ptr<RColumnReaderBase<T>> &colReader, unsigned int slot,
                             RDFDetail::RCustomColumnBase *customCol, TTreeReader *r, const std::string &colName)
{
   R__ASSERT(customCol != nullptr || r != nullptr);
   auto newColReader = MakeColumnReader<T>(slot, customCol, r, colName);
   colReader.swap(newColReader);
}

/// Initialize a tuple of column readers.
/// For real TTree branches a TTreeReader{Array,Value} is built and passed to the
/// RColumnValue. For temporary columns a pointer to the corresponding variable
/// is passed instead.
template <typename RDFValueTuple, std::size_t... S>
void InitColumnReaders(unsigned int slot, RDFValueTuple &valueTuple, TTreeReader *r,
                       const std::vector<std::string> &colNames, const RBookedCustomColumns &customCols,
                       std::index_sequence<S...>, const std::array<bool, sizeof...(S)> &isCustomColumn)
{
   const auto &customColMap = customCols.GetColumns();

   using expander = int[];

   // Hack to expand a parameter pack without c++17 fold expressions.
   // Construct the column readers
   (void)expander{
      (InitColumnReadersHelper(std::get<S>(valueTuple), slot,
                               isCustomColumn[S] ? customColMap.at(colNames[S]).get() : nullptr, r, colNames[S]),
       0)...,
      0};

   (void)slot;     // avoid _bogus_ "unused variable" warnings for slot on gcc 4.9
   (void)r;        // avoid "unused variable" warnings for r on gcc5.2
}

/// Clear the proxies of a tuple of RColumnValues
template <typename ValueTuple, std::size_t... S>
void ResetColumnReaders(ValueTuple &values, std::index_sequence<S...>)
{
   // hack to expand a parameter pack without c++17 fold expressions.
   int expander[] = {(std::get<S>(values)->Reset(), 0)..., 0};
   (void)expander; // avoid "unused variable" warnings
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_COLUMNREADERS
