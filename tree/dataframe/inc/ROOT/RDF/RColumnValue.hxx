// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RCOLUMNVALUE
#define ROOT_RCOLUMNVALUE

#include <ROOT/RDF/RCustomColumnBase.hxx>
#include <ROOT/RDF/Utils.hxx> // IsRVec_t, TypeID2TypeName
#include <ROOT/RIntegerSequence.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/TypeTraits.hxx> // TakeFirstParameter_t
#include <RtypesCore.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <cstring> // strcmp
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {
using namespace ROOT::VecOps;

/**
\class ROOT::Internal::RDF::RColumnValue
\ingroup dataframe
\brief Helper class that updates and returns TTree branches as well as RDataFrame temporary columns
\tparam T The type of the column

RDataFrame nodes must access two different types of values during the event loop:
values of real branches, for which TTreeReader{Values,Arrays} act as proxies, or
temporary columns whose values are generated on the fly. While the type of the
value is known at compile time (or just-in-time), it is only at runtime that nodes
can check whether a certain value is generated on the fly or not.

RColumnValue abstracts this difference by providing the same interface for
both cases and handling the reading or generation of new values transparently.
Only one of the two data members fReaderProxy or fValuePtr will be non-null
for a given RColumnValue, depending on whether the value comes from a real
TTree branch or from a temporary column respectively.

RDataFrame nodes can store tuples of RColumnValues and retrieve an updated
value for the column via the `Get` method.
**/
template <typename T>
class R__CLING_PTRCHECK(off) RColumnValue {
// R__CLING_PTRCHECK is disabled because all pointers are hand-crafted by RDF.

   using MustUseRVec_t = IsRVec_t<T>;

   // ColumnValue_t is the type of the column or the type of the elements of an array column
   using ColumnValue_t = typename std::conditional<MustUseRVec_t::value, TakeFirstParameter_t<T>, T>::type;
   using TreeReader_t = typename std::conditional<MustUseRVec_t::value, TTreeReaderArray<ColumnValue_t>,
                                                  TTreeReaderValue<ColumnValue_t>>::type;

   /// RColumnValue has a slightly different behaviour whether the column comes from a TTreeReader, a RDataFrame Define
   /// or a RDataSource. It stores which it is as an enum.
   enum class EColumnKind { kTree, kCustomColumn, kDataSource, kInvalid };
   // Set to the correct value by MakeProxy or SetTmpColumn
   EColumnKind fColumnKind = EColumnKind::kInvalid;
   /// The slot this value belongs to. Only needed when querying custom column values, it is set in `SetTmpColumn`.
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();

   // Each element of the following stacks will be in use by a _single task_.
   // Each task will push one element when it starts and pop it when it ends.
   // Stacks will typically be very small (1-2 elements typically) and will only grow over size 1 in case of interleaved
   // task execution i.e. when more than one task needs readers in this worker thread.

   /// Owning ptrs to a TTreeReaderValue or TTreeReaderArray. Only used for Tree columns.
   std::unique_ptr<TreeReader_t> fTreeReader;
   /// Non-owning ptrs to the value of a custom column.
   T *fCustomValuePtr;
   /// Non-owning ptrs to the value of a data-source column.
   T **fDSValuePtr;
   /// Non-owning ptrs to the node responsible for the custom column. Needed when querying custom values.
   RCustomColumnBase *fCustomColumn;
   /// Enumerator for the different properties of the branch storage in memory
   enum class EStorageType : char { kContiguous, kUnknown, kSparse };
   /// Signal whether we ever checked that the branch we are reading with a TTreeReaderArray stores array elements
   /// in contiguous memory. Only used when T == RVec<U>.
   EStorageType fStorageType = EStorageType::kUnknown;
   /// If MustUseRVec, i.e. we are reading an array, we return a reference to this RVec to clients
   RVec<ColumnValue_t> fRVec;
   bool fCopyWarningPrinted = false;

public:
   RColumnValue(){};

   void SetTmpColumn(unsigned int slot, RCustomColumnBase *customColumn)
   {
      fCustomColumn = customColumn;
      // Here we compare names and not typeinfos since they may come from two different contexts: a compiled
      // and a jitted one.
      const auto diffTypes = (0 != std::strcmp(customColumn->GetTypeId().name(), typeid(T).name()));
      auto inheritedType = [&](){
         auto colTClass = TClass::GetClass(customColumn->GetTypeId());
         return colTClass && colTClass->InheritsFrom(TClass::GetClass<T>());
      };

      if (diffTypes && !inheritedType()) {
         const auto tName = TypeID2TypeName(typeid(T));
         const auto colTypeName = TypeID2TypeName(customColumn->GetTypeId());
         std::string errMsg = "RColumnValue: type specified for column \"" +
                              customColumn->GetName() + "\" is ";
         if (tName.empty()) {
            errMsg += typeid(T).name();
            errMsg += " (extracted from type info)";
         } else {
            errMsg += tName;
         }
         errMsg += " but temporary column has type ";
         if (colTypeName.empty()) {
            auto &id = customColumn->GetTypeId();
            errMsg += id.name();
            errMsg += " (extracted from type info)";
         } else {
            errMsg += colTypeName;
         }
         throw std::runtime_error(errMsg);
      }

      if (customColumn->IsDataSourceColumn()) {
         fColumnKind = EColumnKind::kDataSource;
         fDSValuePtr = static_cast<T **>(customColumn->GetValuePtr(slot));
      } else {
         fColumnKind = EColumnKind::kCustomColumn;
         fCustomValuePtr = static_cast<T *>(customColumn->GetValuePtr(slot));
      }
      fSlot = slot;
   }

   void MakeProxy(TTreeReader *r, const std::string &bn)
   {
      fColumnKind = EColumnKind::kTree;
      fTreeReader = std::make_unique<TreeReader_t>(*r, bn.c_str());
   }

   /// This overload is used to return scalar quantities (i.e. types that are not read into a RVec)
   // This method is executed inside the event-loop, many times per entry
   // If need be, the if statement can be avoided using thunks
   // (have both branches inside functions and have a pointer to the branch to be executed)
   template <typename U = T, typename std::enable_if<!RColumnValue<U>::MustUseRVec_t::value, int>::type = 0>
   T &Get(Long64_t entry)
   {
      if (fColumnKind == EColumnKind::kTree) {
         return *(fTreeReader->Get());
      } else {
         fCustomColumn->Update(fSlot, entry);
         return fColumnKind == EColumnKind::kCustomColumn ? *fCustomValuePtr : **fDSValuePtr;
      }
   }

   /// This overload is used to return arrays (i.e. types that are read into a RVec).
   /// In this case the returned T is always a RVec<ColumnValue_t>.
   /// RVec<bool> is treated differently, in a separate overload.
   template <typename U = T,
             typename std::enable_if<RColumnValue<U>::MustUseRVec_t::value && !std::is_same<U, RVec<bool>>::value,
                                     int>::type = 0>
   T &Get(Long64_t entry)
   {
      if (fColumnKind == EColumnKind::kTree) {
         auto &readerArray = *fTreeReader;
         // We only use TTreeReaderArrays to read columns that users flagged as type `RVec`, so we need to check
         // that the branch stores the array as contiguous memory that we can actually wrap in an `RVec`.
         // Currently we need the first entry to have been loaded to perform the check
         // TODO Move check to `MakeProxy` once Axel implements this kind of check in TTreeReaderArray using
         // TBranchProxy

         const auto arrSize = readerArray.GetSize();
         if (EStorageType::kUnknown == fStorageType && arrSize > 1) {
            // We can decide since the array is long enough
            fStorageType = EStorageType::kContiguous;
            for (auto i = 0u; i < arrSize - 1; ++i) {
               if ((char *)&readerArray[i + 1] - (char *)&readerArray[i] != sizeof(typename U::value_type)) {
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
               T rvec(readerArrayAddr, readerArraySize);
               std::swap(fRVec, rvec);
            } else {
               T emptyVec{};
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
               T rvec(readerArray.begin(), readerArray.end());
               std::swap(fRVec, rvec);
            } else {
               T emptyVec{};
               std::swap(fRVec, emptyVec);
            }
         }
         return fRVec;

      } else {
         fCustomColumn->Update(fSlot, entry);
         return fColumnKind == EColumnKind::kCustomColumn ? *fCustomValuePtr : **fDSValuePtr;
      }
   }

   /// This overload covers the RVec<bool> case. In this case we always copy the contents of TTreeReaderArray<bool>
   /// into RVec<bool> (never take a view into the memory buffer) because the underlying memory buffer might be the
   /// one of a std::vector<bool>, which is not a contiguous slab of bool values.
   /// Note that this also penalizes the case in which the column type is actually bool[], but the possible performance
   /// gains in this edge case is probably not worth the extra complication required to differentiate the two cases.
   template <typename U = T,
             typename std::enable_if<RColumnValue<U>::MustUseRVec_t::value && std::is_same<U, RVec<bool>>::value,
                                     int>::type = 0>
   T &Get(Long64_t entry)
   {
      if (fColumnKind == EColumnKind::kTree) {
         auto &readerArray = *fTreeReader;
         const auto readerArraySize = readerArray.GetSize();
         if (readerArraySize > 0) {
            // always perform a copy
            T rvec(readerArray.begin(), readerArray.end());
            std::swap(fRVec, rvec);
         } else {
            T emptyVec{};
            std::swap(fRVec, emptyVec);
         }
         return fRVec;
      } else {
         // business as usual
         fCustomColumn->Update(fSlot, entry);
         return fColumnKind == EColumnKind::kCustomColumn ? *fCustomValuePtr : **fDSValuePtr;
      }
   }

   void Reset()
   {
      // This method should by all means not be removed, together with all
      // of its callers, otherwise a race condition takes place in which a
      // TTreeReader and its TTreeReader{Value,Array}s could be deleted
      // concurrently:
      // - Thread #1) a task ends and pushes back processing slot
      // - Thread #2) a task starts and overwrites thread-local TTreeReaderValues
      // - Thread #1) first task deletes TTreeReader
      // See https://github.com/root-project/root/commit/26e8ace6e47de6794ac9ec770c3bbff9b7f2e945
      if (EColumnKind::kTree == fColumnKind) {
         fTreeReader.reset();
      }
   }
};

// Some extern instantiations to speed-up compilation/interpretation time
// These are not active if c++17 is enabled because of a bug in our clang
// See ROOT-9499.
#if __cplusplus < 201703L
extern template class RColumnValue<int>;
extern template class RColumnValue<unsigned int>;
extern template class RColumnValue<char>;
extern template class RColumnValue<unsigned char>;
extern template class RColumnValue<float>;
extern template class RColumnValue<double>;
extern template class RColumnValue<Long64_t>;
extern template class RColumnValue<ULong64_t>;
extern template class RColumnValue<std::vector<int>>;
extern template class RColumnValue<std::vector<unsigned int>>;
extern template class RColumnValue<std::vector<char>>;
extern template class RColumnValue<std::vector<unsigned char>>;
extern template class RColumnValue<std::vector<float>>;
extern template class RColumnValue<std::vector<double>>;
extern template class RColumnValue<std::vector<Long64_t>>;
extern template class RColumnValue<std::vector<ULong64_t>>;
#endif

template <typename T>
struct RDFValueTuple {
};

template <typename... BranchTypes>
struct RDFValueTuple<TypeList<BranchTypes...>> {
   using type = std::tuple<RColumnValue<BranchTypes>...>;
};

template <typename BranchType>
using RDFValueTuple_t = typename RDFValueTuple<BranchType>::type;

/// Clear the proxies of a tuple of RColumnValues
template <typename ValueTuple, std::size_t... S>
void ResetRDFValueTuple(ValueTuple &values, std::index_sequence<S...>)
{
   // hack to expand a parameter pack without c++17 fold expressions.
   std::initializer_list<int> expander{(std::get<S>(values).Reset(), 0)...};
   (void)expander; // avoid "unused variable" warnings
}


} // ns RDF
} // ns Internal
} // ns ROOT

#endif // ROOT_RCOLUMNVALUE
