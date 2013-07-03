// @(#)root/tree:$Id$
// Author: Axel Naumann, 2010-08-02

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
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

#ifdef __CINT__
#pragma link C++ class TTreeReaderValue<Int_t>+; // For the automatic array size reader
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

      TCollectionReaderABC* fImpl; // Common interface to collections

      ClassDefT(TTreeReaderArrayBase, 0);//Accessor to member of an object stored in a collection
   };

} // namespace ROOT

template <typename T>
class TTreeReaderArray: public ROOT::TTreeReaderArrayBase {
public:
   TTreeReaderArray(TTreeReader& tr, const char* branchname):
      TTreeReaderArrayBase(&tr, branchname, TDictionary::GetDictionary(typeid(T)))
   {
      // Create an array reader of branch "branchname" for TTreeReader "tr".
   }

   T& At(size_t idx) { return *(T*)UntypedAt(idx); }
   T& operator[](size_t idx) { return At(idx); }

   ClassDefT(TTreeReaderArray, 0);//Accessor to member of an object stored in a collection
};

#endif // ROOT_TTreeReaderArray
