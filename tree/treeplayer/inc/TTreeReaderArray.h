// @(#)root/tree:$Id$
// Author: Axel Naumann, 2010-08-02

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeReaderArray
#define ROOT_TTreeReaderArray




#include "TTreeReaderValue.h"
#include "TTreeReaderUtils.h"
#include <type_traits>

namespace ROOT {
namespace Internal {

/** \class TTreeReaderArrayBase
Base class of TTreeReaderArray.
*/

   class TTreeReaderArrayBase: public TTreeReaderValueBase {
   public:
      TTreeReaderArrayBase(TTreeReader* reader, const char* branchname,
                           TDictionary* dict):
         TTreeReaderValueBase(reader, branchname, dict) {}

      std::size_t GetSize() const { return fImpl->GetSize(GetProxy()); }
      Bool_t IsEmpty() const { return !GetSize(); }

      virtual EReadStatus GetReadStatus() const { return fImpl ? fImpl->fReadStatus : kReadError; }

   protected:
      void *UntypedAt(std::size_t idx) const { return fImpl->At(GetProxy(), idx); }
      virtual void CreateProxy();
      bool GetBranchAndLeaf(TBranch* &branch, TLeaf* &myLeaf,
                            TDictionary* &branchActualType);
      void SetImpl(TBranch* branch, TLeaf* myLeaf);
      const char* GetBranchContentDataType(TBranch* branch,
                                           TString& contentTypeName,
                                           TDictionary* &dict);

      std::unique_ptr<TVirtualCollectionReader> fImpl; // Common interface to collections

      // FIXME: re-introduce once we have ClassDefInline!
      //ClassDef(TTreeReaderArrayBase, 0);//Accessor to member of an object stored in a collection
   };

} // namespace Internal
} // namespace ROOT

// clang-format off
/**
 * \class TTreeReaderArray
 * \ingroup treeplayer
 * \brief An interface for reading collections stored in ROOT columnar datasets
 *
 * The TTreeReaderArray is a type-safe tool to be used in association with a TTreeReader
 * to access the collections stored in TTree, TNtuple and TChain datasets.
 * In order to access values which are not collections, the TTreeReaderValue class can
 * be used.
 *
 * See the documentation of TTreeReader for more details and examples.
*/
// clang-format on

template <typename T>
class R__CLING_PTRCHECK(off) TTreeReaderArray final : public ROOT::Internal::TTreeReaderArrayBase {
// R__CLING_PTRCHECK is disabled because pointer / types are checked by CreateProxy().

public:
   /// Random access iterator to the elements of a TTreeReaderArray.
   // The template parameter is there to allow distinguishing between the `const` and `non-const` cases.
   template <typename ReaderArrayType>
   class Iterator_t {
   public:
      // iterators must define the following types
      using iterator_category = std::random_access_iterator_tag;
      using value_type = T;
      using difference_type = std::ptrdiff_t;
      using pointer = std::conditional_t<std::is_const<ReaderArrayType>::value, const T *, T *>;
      using reference = std::conditional_t<std::is_const<ReaderArrayType>::value, const T &, T &>;

   private:
      TTreeReaderArray *fArray; ///< The array iterated over; nullptr if invalid/past-the-end.
      std::size_t fIndex;       ///< Current index in the array.
      std::size_t fSize;        ///< Size of the TTreeReaderArray
   public:
      /// Default ctor: constructs a past-the-end iterator
      Iterator_t() : fArray(nullptr), fIndex(0u), fSize(0u) {}

      /// Construct iterator
      Iterator_t(std::size_t index, TTreeReaderArray *array)
         : fArray(array), fIndex(index), fSize(fArray ? fArray->GetSize() : 0u)
      {
         if (fIndex >= fSize)
            fArray = nullptr; // invalidate iterator
      }

      /// Construct iterator from a const TTreeReaderArray
      Iterator_t(std::size_t index, const TTreeReaderArray *array)
         : Iterator_t(index, const_cast<TTreeReaderArray *>(array)) {}

      Iterator_t(Iterator_t &&) = default;
      Iterator_t(const Iterator_t &) = default;
      Iterator_t &operator=(Iterator_t &&) = default;
      Iterator_t &operator=(const Iterator_t &) = default;

      reference operator*() const
      {
         R__ASSERT(fArray && "invalid iterator!");
         return fArray->At(fIndex);
      }

      pointer operator->() const { return IsValid() ? &fArray->At(fIndex) : nullptr; }

      bool operator==(const Iterator_t &other) const
      {
         // Follow C++14 requiring two past-the-end iterators to be equal.
         if (!IsValid() && !other.IsValid())
            return true;
         return fArray == other.fArray && fIndex == other.fIndex;
      }

      bool operator!=(const Iterator_t &other) const { return !(*this == other); }

      /// Pre-increment operator
      Iterator_t &operator++()
      {
         if (IsValid())
            ++fIndex;
         if (fIndex >= fSize)
            fArray = nullptr; // invalidate iterator
         return *this;
      }

      /// Post-increment operator
      Iterator_t operator++(int)
      {
         auto ret = *this;
         this->operator++();
         return ret;
      }

      /// Pre-decrement operator
      Iterator_t &operator--()
      {
         if (fIndex == 0u)
            fArray = nullptr; // invalidate iterator
         else
            --fIndex;
         return *this;
      }

      /// Post-decrement operator
      Iterator_t operator--(int)
      {
         auto ret = *this;
         this->operator--();
         return ret;
      }

      Iterator_t operator+(std::ptrdiff_t n) const { return Iterator_t(fIndex + n, fArray); }
      friend auto operator+(std::ptrdiff_t n, const Iterator_t &it) -> decltype(it + n) { return it + n; }

      Iterator_t operator-(std::ptrdiff_t n) const
      {
         const auto index = std::ptrdiff_t(fIndex);
         const auto newIndex = index >= n ? index - n : std::numeric_limits<decltype(fIndex)>::max();
         return Iterator_t(newIndex, fArray);
      }

      std::ptrdiff_t operator-(const Iterator_t &other) const { return fIndex - other.fIndex; }

      Iterator_t &operator+=(std::ptrdiff_t n) { return (*this = *this + n); }

      Iterator_t &operator-=(std::ptrdiff_t n) { return (*this = *this - n); }

      bool operator<(const Iterator_t &other) const { return fIndex < other.fIndex; }
      bool operator>(const Iterator_t &other) const { return fIndex > other.fIndex; }
      bool operator<=(const Iterator_t &other) const { return !(*this > other); }
      bool operator>=(const Iterator_t &other) const { return !(*this < other); }

      reference operator[](std::size_t index) const { return *(*this + index); }

      operator pointer() { return &fArray->At(fIndex); }

      bool IsValid() const { return fArray != nullptr; }
   };

   using iterator = Iterator_t<TTreeReaderArray<T>>;
   using const_iterator = Iterator_t<const TTreeReaderArray<T>>;

   /// Create an array reader of branch "branchname" for TTreeReader "tr".
   TTreeReaderArray(TTreeReader &tr, const char *branchname)
      : TTreeReaderArrayBase(&tr, branchname, TDictionary::GetDictionary(typeid(T))) {}

   T &At(std::size_t idx) { return *static_cast<T *>(UntypedAt(idx)); }
   const T &At(std::size_t idx) const { return *static_cast<T *>(UntypedAt(idx)); }
   T &operator[](std::size_t idx) { return At(idx); }
   const T &operator[](std::size_t idx) const { return At(idx); }

   iterator begin() { return iterator(0u, this); }
   iterator end() { return iterator(GetSize(), this); }
   const_iterator begin() const { return cbegin(); }
   const_iterator end() const { return cend(); }
   const_iterator cbegin() const { return const_iterator(0u, this); }
   const_iterator cend() const { return const_iterator(GetSize(), this); }

protected:
#define R__TTreeReaderArray_TypeString(T) #T
   virtual const char *GetDerivedTypeName() const { return R__TTreeReaderArray_TypeString(T); }
#undef R__TTreeReaderArray_TypeString
   // FIXME: re-introduce once we have ClassDefTInline!
   // ClassDefT(TTreeReaderArray, 0);//Accessor to member of an object stored in a collection
};

#endif // ROOT_TTreeReaderArray
