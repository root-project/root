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


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTreeReaderArray                                                    //
//                                                                        //
// A simple interface for reading data from trees or chains.              //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTreeReaderValue
#include "TTreeReaderValue.h"
#endif
#ifndef ROOT_TTreeReaderUtils
#include "TTreeReaderUtils.h"
#endif

namespace ROOT {
   class TTreeReaderArrayBase: public TTreeReaderValueBase {
   public:
      TTreeReaderArrayBase(TTreeReader* reader, const char* branchname,
                           TDictionary* dict):
         TTreeReaderValueBase(reader, branchname, dict), fImpl(0) {}

      size_t GetSize() const { return fImpl->GetSize(GetProxy()); }
      Bool_t IsEmpty() const { return !GetSize(); }

      virtual EReadStatus GetReadStatus() const { return fImpl ? fImpl->fReadStatus : kReadError; }

   protected:
      void* UntypedAt(size_t idx) const { return fImpl->At(GetProxy(), idx); }
      virtual void CreateProxy();
      const char* GetBranchContentDataType(TBranch* branch,
                                           TString& contentTypeName,
                                           TDictionary* &dict) const;

      TVirtualCollectionReader* fImpl; // Common interface to collections

      // FIXME: re-introduce once we have ClassDefInline!
      //ClassDef(TTreeReaderArrayBase, 0);//Accessor to member of an object stored in a collection
   };

} // namespace ROOT

template <typename T>
class TTreeReaderArray: public ROOT::TTreeReaderArrayBase {
public:

   // Iterator through the indices of a TTreeReaderArray.
   struct Iterator_t:
      public std::iterator<std::input_iterator_tag, T, long> {
      // Default initialized, past-end iterator.
      Iterator_t() :
         fIndex(0), fArray(0) {}

      // Initialize with an array and index.
      Iterator_t(size_t idx, TTreeReaderArray* array) :
         fIndex(idx), fArray(array) {}

      size_t fIndex; // Current index in the array.
      TTreeReaderArray* fArray; // The array iterated over; 0 if invalid / end.

      bool IsValid() const { return fArray; }

      bool operator==(const Iterator_t& lhs) const {
         // Compare two iterators as equal; follow C++14 requiring two past-end
         // iterators to be equal.
         if (!IsValid() && !lhs.IsValid())
            return true;
         return fIndex == lhs.fIndex && fArray == lhs.fArray;
      }

      bool operator!=(const Iterator_t& lhs) const {
         // Compare not equal.
         return !(*this == lhs);
      }

      Iterator_t operator++(int) {
         // Post-increment (it++).
         Iterator_t ret = *this;
         this->operator++();
         return ret;
      }

      Iterator_t& operator++() {
         // Pre-increment (++it).
         if (IsValid()) {
            ++fIndex;
            if (fIndex >= fArray->GetSize()) {
               // Remember that it's past-end.
               fArray = 0;
            }
         }
         return *this;
      }

      T& operator*() const {
         // Get the referenced element.
         R__ASSERT(fArray && "invalid iterator!");
         return fArray->At(fIndex);
      }
   };

   typedef Iterator_t iterator;

   TTreeReaderArray(TTreeReader& tr, const char* branchname):
      TTreeReaderArrayBase(&tr, branchname, TDictionary::GetDictionary(typeid(T)))
   {
      // Create an array reader of branch "branchname" for TTreeReader "tr".
   }

   T& At(size_t idx) { return *(T*)UntypedAt(idx); }
   T& operator[](size_t idx) { return At(idx); }

   Iterator_t begin() {
      // Return an iterator to the 0th TTree entry or an empty iterator if the
      // array is empty.
      return IsEmpty() ? Iterator_t() : Iterator_t(0, this);
   }
   Iterator_t end() const { return Iterator_t(); }

protected:
#define R__TTreeReaderArray_TypeString(T) #T
   virtual const char* GetDerivedTypeName() const { return R__TTreeReaderArray_TypeString(T); }
#undef R__TTreeReaderArray_TypeString
   // FIXME: re-introduce once we have ClassDefTInline!
   //ClassDefT(TTreeReaderArray, 0);//Accessor to member of an object stored in a collection
};

#endif // ROOT_TTreeReaderArray
