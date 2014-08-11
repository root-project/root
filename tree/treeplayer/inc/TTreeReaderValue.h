// @(#)root/tree:$Id$
// Author: Axel Naumann, 2010-08-02

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeReaderValue
#define ROOT_TTreeReaderValue


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTreeReaderValue                                                    //
//                                                                        //
// A simple interface for reading data from trees or chains.              //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif
#ifndef ROOT_TBranchProxy
#include "TBranchProxy.h"
#endif

class TBranch;
class TBranchElement;
class TLeaf;
class TTreeReader;

namespace ROOT {

   class TTreeReaderValueBase {
   public:

      // Status flags, 0 is good
      enum ESetupStatus {
         kSetupNotSetup = -7,
         kSetupTreeDestructed = -8,
         kSetupMakeClassModeMismatch = -7, // readers disagree on whether TTree::SetMakeBranch() should be on
         kSetupMissingCounterBranch = -6,
         kSetupMissingBranch = -5,
         kSetupInternalError = -4,
         kSetupMissingCompiledCollectionProxy = -3,
         kSetupMismatch = -2,
         kSetupClassMismatch = -1,
         kSetupMatch = 0,
         kSetupMatchBranch = 0,
         kSetupMatchConversion,
         kSetupMatchConversionCollection,
         kSetupMakeClass,
         kSetupVoidPtr,
         kSetupNoCheck,
         kSetupMatchLeaf
      };
      enum EReadStatus {
         kReadSuccess = 0, // data read okay
         kReadNothingYet, // data now yet accessed
         kReadError // problem reading data
      };

      EReadStatus ProxyRead();

      Bool_t IsValid() const { return fProxy && 0 == (int)fSetupStatus && 0 == (int)fReadStatus; }
      ESetupStatus GetSetupStatus() const { return fSetupStatus; }
      virtual EReadStatus GetReadStatus() const { return fReadStatus; }

      TLeaf* GetLeaf();

      void* GetAddress();

      const char* GetBranchName() const { return fBranchName; }

   protected:
      TTreeReaderValueBase(TTreeReader* reader = 0, const char* branchname = 0, TDictionary* dict = 0);

      virtual ~TTreeReaderValueBase();

      virtual void CreateProxy();
      const char* GetBranchDataType(TBranch* branch,
                                    TDictionary* &dict) const;

      virtual const char* GetDerivedTypeName() const = 0;

      ROOT::TBranchProxy* GetProxy() const { return fProxy; }

      void MarkTreeReaderUnavailable() { fTreeReader = 0; }

      TString      fBranchName; // name of the branch to read data from.
      TString      fLeafName;
      TTreeReader* fTreeReader; // tree reader we belong to
      TDictionary* fDict; // type that the branch should contain
      ROOT::TBranchProxy* fProxy; // proxy for this branch, owned by TTreeReader
      TLeaf*       fLeaf;
      Long64_t     fTreeLastOffset;
      ESetupStatus fSetupStatus; // setup status of this data access
      EReadStatus  fReadStatus; // read status of this data access
      std::vector<Long64_t> fStaticClassOffsets;

      // FIXME: re-introduce once we have ClassDefInline!
      //ClassDef(TTreeReaderValueBase, 0);//Base class for accessors to data via TTreeReader

      friend class ::TTreeReader;
   };

} // namespace ROOT


template <typename T>
class TTreeReaderValue: public ROOT::TTreeReaderValueBase {
public:
   TTreeReaderValue() {}
   TTreeReaderValue(TTreeReader& tr, const char* branchname):
      TTreeReaderValueBase(&tr, branchname, TDictionary::GetDictionary(typeid(T))) {}

   T* Get() {
      if (!fProxy){
         Error("Get()", "Value reader not properly initialized, did you remember to call TTreeReader.Set(Next)Entry()?");
         return 0;
      }
      void *address = GetAddress(); // Needed to figure out if it's a pointer
      return fProxy->IsaPointer() ? *(T**)address : (T*)address; }
   T* operator->() { return Get(); }
   T& operator*() { return *Get(); }

protected:
   // FIXME: use IsA() instead once we have ClassDefTInline
#define R__TTreeReaderValue_TypeString(T) #T
   virtual const char* GetDerivedTypeName() const { return R__TTreeReaderValue_TypeString(T); }
#undef R__TTreeReaderValue_TypeString

   // FIXME: re-introduce once we have ClassDefTInline!
   //ClassDefT(TTreeReaderValue, 0);//Accessor to data via TTreeReader
};

#endif // ROOT_TTreeReaderValue
