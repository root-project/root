// @(#)root/tree:$Id$
// Author: Axel Naumann, 2010-08-02
// Author: Vincenzo Eduardo Padulano CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeReaderValue
#define ROOT_TTreeReaderValue

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TTreeReaderValue                                                       //
//                                                                        //
// A simple interface for reading data from trees or chains.              //
//                                                                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include "TDictionary.h"
#include "TBranchProxy.h"

#include <type_traits>
#include <vector>
#include <string>

class TBranch;
class TBranchElement;
class TLeaf;
class TTreeReader;

namespace ROOT {
namespace Internal {

/** \class TTreeReaderValueBase
Base class of TTreeReaderValue.
*/

class TTreeReaderValueBase {
public:
   /// Status flags, 0 is good
   enum ESetupStatus {
      kSetupNotSetup = -7,              ///< No initialization has happened yet.
      kSetupTreeDestructed = -8,        ///< The TTreeReader has been destructed / not set.
      kSetupMakeClassModeMismatch = -9, ///< readers disagree on whether TTree::SetMakeBranch() should be on
      kSetupMissingCounterBranch = -6,  ///< The array cannot find its counter branch: Array[CounterBranch]
      kSetupMissingBranch = -5,         ///< The specified branch cannot be found.
      kSetupInternalError = -4,         ///< Some other error - hopefully the error message helps.
      kSetupMissingDictionary = -3,     ///< To read this branch, we need a dictionary.
      kSetupMismatch = -2,              ///< Mismatch of branch type and reader template type.
      kSetupNotACollection = -1,        ///< The branch class type is not a collection.
      kSetupMatch =
         0, ///< This branch has been set up, branch data type and reader template type match, reading should succeed.
      kSetupMatchBranch =
         7, ///< This branch has been set up, branch data type and reader template type match, reading should succeed.
      // kSetupMatchConversion = 1, /// This branch has been set up, the branch data type can be converted to the reader
      // template type, reading should succeed. kSetupMatchConversionCollection = 2, /// This branch has been set up,
      // the data type of the branch's collection elements can be converted to the reader template type, reading should
      // succeed. kSetupMakeClass = 3, /// This branch has been set up, enabling MakeClass mode for it, reading should
      // succeed.
      //  kSetupVoidPtr = 4,
      kSetupNoCheck = 5,
      kSetupMatchLeaf = 6 ///< This branch (or TLeaf, really) has been set up, reading should succeed.
   };
   enum EReadStatus {
      kReadSuccess = 0, ///< Data read okay
      kReadNothingYet,  ///< Data now yet accessed
      kReadError        ///< Problem reading data
   };

   EReadStatus ProxyRead() { return (this->*fProxyReadFunc)(); }

   EReadStatus ProxyReadDefaultImpl();

   typedef bool (ROOT::Detail::TBranchProxy::*BranchProxyRead_t)();
   template <BranchProxyRead_t Func>
   ROOT::Internal::TTreeReaderValueBase::EReadStatus ProxyReadTemplate();

   /// Return true if the branch was setup \em and \em read correctly.
   /// Use GetSetupStatus() to only check the setup status.
   bool IsValid() const { return fProxy && 0 == (int)fSetupStatus && 0 == (int)fReadStatus; }
   /// Return this TTreeReaderValue's setup status.
   /// Use this method to check e.g. whether the TTreeReaderValue is correctly setup and ready for reading.
   ESetupStatus GetSetupStatus() const { return fSetupStatus; }
   virtual EReadStatus GetReadStatus() const { return fReadStatus; }

   /// If we are reading a leaf, return the corresponding TLeaf.
   TLeaf *GetLeaf() { return fLeaf; }

   void *GetAddress();

   const char *GetBranchName() const { return fBranchName; }

   virtual ~TTreeReaderValueBase();

protected:
   TTreeReaderValueBase(TTreeReader *reader, const char *branchname, TDictionary *dict, bool opaqueRead = false);
   TTreeReaderValueBase(const TTreeReaderValueBase &);
   TTreeReaderValueBase &operator=(const TTreeReaderValueBase &);

   void RegisterWithTreeReader();
   void NotifyNewTree(TTree *newTree);

   TBranch *SearchBranchWithCompositeName(TLeaf *&myleaf, TDictionary *&branchActualType, std::string &err);
   virtual void CreateProxy();
   static const char *GetBranchDataType(TBranch *branch, TDictionary *&dict, TDictionary const *curDict);

   virtual const char *GetDerivedTypeName() const = 0;

   Detail::TBranchProxy *GetProxy() const { return fProxy; }

   void MarkTreeReaderUnavailable()
   {
      fTreeReader = nullptr;
      fSetupStatus = kSetupTreeDestructed;
   }

   /// Stringify the template argument.
   static std::string GetElementTypeName(const std::type_info &ti);

   void ErrorAboutMissingProxyIfNeeded();

   bool fHaveLeaf : 1;                         ///< Whether the data is in a leaf
   bool fHaveStaticClassOffsets : 1;           ///< Whether !fStaticClassOffsets.empty()
   EReadStatus fReadStatus : 2;                ///< Read status of this data access
   ESetupStatus fSetupStatus = kSetupNotSetup; ///< Setup status of this data access
   TString fBranchName;                        ///< Name of the branch to read data from.
   TString fLeafName;
   TTreeReader *fTreeReader;               ///< Tree reader we belong to
   TDictionary *fDict;                     ///< Type that the branch should contain
   Detail::TBranchProxy *fProxy = nullptr; ///< Proxy for this branch, owned by TTreeReader
   TLeaf *fLeaf = nullptr;
   std::vector<Long64_t> fStaticClassOffsets;
   typedef EReadStatus (TTreeReaderValueBase::*Read_t)();
   Read_t fProxyReadFunc = &TTreeReaderValueBase::ProxyReadDefaultImpl; ///<! Pointer to the Read implementation to use.
   /**
    * If true, the reader will not do any type-checking against the actual
    * type held by the branch. Useful to just check if the current entry can
    * be read or not without caring about its value.
    * \note Only used by TTreeReaderOpaqueValue.
    */
   bool fOpaqueRead{false};

   // FIXME: re-introduce once we have ClassDefInline!
   // ClassDefOverride(TTreeReaderValueBase, 0);//Base class for accessors to data via TTreeReader

   friend class ::TTreeReader;
};

/**
 * \brief Read a value in a branch without knowledge of its type
 *
 * This class is helpful in situations where the actual contents of the branch
 * at the current entry are not relevant and one only wants to know whether
 * the entry can be read.
 */
class R__CLING_PTRCHECK(off) TTreeReaderOpaqueValue final : public ROOT::Internal::TTreeReaderValueBase {
public:
   TTreeReaderOpaqueValue(TTreeReader &tr, const char *branchname)
      : TTreeReaderValueBase(&tr, branchname, /*dict*/ nullptr, /*opaqueRead*/ true)
   {
   }

protected:
   const char *GetDerivedTypeName() const { return ""; }
};

class R__CLING_PTRCHECK(off) TTreeReaderUntypedValue final : public TTreeReaderValueBase {
   std::string fElementTypeName;

public:
   TTreeReaderUntypedValue(TTreeReader &tr, std::string_view branchName, std::string_view typeName)
      : TTreeReaderValueBase(&tr, branchName.data(), TDictionary::GetDictionary(typeName.data())),
        fElementTypeName(typeName)
   {
   }

   void *Get()
   {
      if (!fProxy) {
         ErrorAboutMissingProxyIfNeeded();
         return nullptr;
      }
      void *address = GetAddress(); // Needed to figure out if it's a pointer
      return fProxy->IsaPointer() ? *(void **)address : (void *)address;
   }

protected:
   const char *GetDerivedTypeName() const final { return fElementTypeName.c_str(); }
};

} // namespace Internal
} // namespace ROOT

template <typename T>
class R__CLING_PTRCHECK(off) TTreeReaderValue final : public ROOT::Internal::TTreeReaderValueBase {
   // R__CLING_PTRCHECK is disabled because pointer / types are checked by CreateProxy().

public:
   using NonConstT_t = typename std::remove_const<T>::type;
   TTreeReaderValue() = delete;
   TTreeReaderValue(TTreeReader &tr, const char *branchname)
      : TTreeReaderValueBase(&tr, branchname, TDictionary::GetDictionary(typeid(NonConstT_t)))
   {
   }

   /// Return a pointer to the value of the current entry.
   /// Return a nullptr and print an error if no entry has been loaded yet.
   /// The returned address is guaranteed to stay constant while a given TTree is being read from a given file,
   /// unless the branch addresses are manipulated directly (e.g. through TTree::SetBranchAddress()).
   /// The address might also change when the underlying TTree/TFile is switched, e.g. when a TChain switches files.
   T *Get()
   {
      if (!fProxy) {
         ErrorAboutMissingProxyIfNeeded();
         return nullptr;
      }
      void *address = GetAddress(); // Needed to figure out if it's a pointer
      return fProxy->IsaPointer() ? *(T **)address : (T *)address;
   }

   /// Return a pointer to the value of the current entry.
   /// Equivalent to Get().
   T *operator->() { return Get(); }

   /// Return a reference to the value of the current entry.
   /// Equivalent to dereferencing the pointer returned by Get(). Behavior is undefined if no entry has been loaded yet.
   /// Most likely a crash will occur.
   T &operator*() { return *Get(); }

protected:
   // FIXME: use IsA() instead once we have ClassDefTInline
   /// Get the template argument as a string.
   const char *GetDerivedTypeName() const override
   {
      static const std::string sElementTypeName = GetElementTypeName(typeid(T));
      return sElementTypeName.data();
   }

   // FIXME: re-introduce once we have ClassDefTInline!
   // ClassDefT(TTreeReaderValue, 0);//Accessor to data via TTreeReader
};

namespace cling {
std::string printValue(ROOT::Internal::TTreeReaderValueBase *val);
template <typename T>
std::string printValue(TTreeReaderValue<T> *val)
{
   return printValue(static_cast<ROOT::Internal::TTreeReaderValueBase *>(val));
}
} // namespace cling

#endif // ROOT_TTreeReaderValue
