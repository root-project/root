// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-28

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReaderArray.h"

#include "TBranchClones.h"
#include "TBranchElement.h"
#include "TBranchRef.h"
#include "TBranchSTL.h"
#include "TBranchObject.h"
#include "TBranchProxyDirector.h"
#include "TClassEdit.h"
#include "TEnum.h"
#include "TFriendElement.h"
#include "TFriendProxy.h"
#include "TLeaf.h"
#include "TList.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TTreeReader.h"
#include "TGenCollectionProxy.h"
#include "TRegexp.h"

#include <memory>
#include <optional>
#include <iostream>

// pin vtable
ROOT::Internal::TVirtualCollectionReader::~TVirtualCollectionReader() {}

namespace {
using namespace ROOT::Internal;

// Reader interface for clones arrays
class TClonesReader : public TVirtualCollectionReader {
public:
   TClonesReader() = default;
   ~TClonesReader() override = default;
   TClonesReader(const TClonesReader &) = delete;
   TClonesReader &operator=(const TClonesReader &) = delete;
   TClonesReader(TClonesReader &&) = delete;
   TClonesReader &operator=(TClonesReader &&) = delete;

   TClonesArray *GetCA(ROOT::Detail::TBranchProxy *proxy)
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TClonesReader::GetCA()", "Read error in TBranchProxy.");
         return nullptr;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return (TClonesArray *)proxy->GetWhere();
   }
   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      TClonesArray *myClonesArray = GetCA(proxy);
      if (myClonesArray) {
         return myClonesArray->GetEntries();
      } else
         return 0;
   }
   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      TClonesArray *myClonesArray = GetCA(proxy);
      if (myClonesArray) {
         return myClonesArray->UncheckedAt(idx);
      } else
         return nullptr;
   }

   bool IsContiguous(ROOT::Detail::TBranchProxy *) override { return false; }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TClonesReader::GetValueSize()", "Read error in TBranchProxy.");
         return 0;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return proxy->GetValueSize();
   }
};

bool IsCPContiguous(const TVirtualCollectionProxy &cp)
{
   if (cp.GetProperties() & TVirtualCollectionProxy::kIsEmulated)
      return true;

   switch (cp.GetCollectionType()) {
   case ROOT::kSTLvector:
   case ROOT::kROOTRVec: return true;
   default: return false;
   }
}

UInt_t GetCPValueSize(const TVirtualCollectionProxy &cp)
{
   // This works only if the collection proxy value type is a fundamental type
   auto &&eDataType = cp.GetType();
   auto *tDataType = TDataType::GetDataType(eDataType);
   return tDataType ? tDataType->Size() : 0;
}

// Reader interface for STL
class TSTLReader final : public TVirtualCollectionReader {
public:
   ~TSTLReader() override {}
   TVirtualCollectionProxy *GetCP(ROOT::Detail::TBranchProxy *proxy)
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TSTLReader::GetCP()", "Read error in TBranchProxy.");
         return nullptr;
      }
      if (!proxy->GetWhere()) {
         Error("TSTLReader::GetCP()", "Logic error, proxy object not set in TBranchProxy.");
         return nullptr;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return (TVirtualCollectionProxy *)proxy->GetCollection();
   }

   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return 0;
      return myCollectionProxy->Size();
   }

   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return nullptr;
      if (myCollectionProxy->HasPointers()) {
         return *(void **)myCollectionProxy->At(idx);
      } else {
         return myCollectionProxy->At(idx);
      }
   }

   bool IsContiguous(ROOT::Detail::TBranchProxy *proxy) override
   {
      auto cp = GetCP(proxy);
      return IsCPContiguous(*cp);
   }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      auto cp = GetCP(proxy);
      return GetCPValueSize(*cp);
   }
};

class TCollectionLessSTLReader final : public TVirtualCollectionReader {
private:
   TVirtualCollectionProxy *fLocalCollection;

public:
   TCollectionLessSTLReader(TVirtualCollectionProxy *proxy) : fLocalCollection(proxy) {}

   TVirtualCollectionProxy *GetCP(ROOT::Detail::TBranchProxy *proxy)
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TCollectionLessSTLReader::GetCP()", "Read error in TBranchProxy.");
         return nullptr;
      }
      if (!proxy->GetWhere()) {
         Error("TCollectionLessSTLReader::GetCP()", "Logic error, proxy object not set in TBranchProxy.");
         return nullptr;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return fLocalCollection;
   }

   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return 0;
      /// In the case of std::vector<bool> `PushProxy` also creates a temporary bool variable the address of which
      /// is returned from these calls.
      myCollectionProxy->PopProxy();
      myCollectionProxy->PushProxy(proxy->GetWhere());
      return myCollectionProxy->Size();
   }

   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return nullptr;
      // Here we do not use a RAII but we empty the proxy to then fill it.
      // This is done because we are returning a pointer and we need to keep
      // alive the memory it points to.
      myCollectionProxy->PopProxy();
      myCollectionProxy->PushProxy(proxy->GetWhere());
      if (myCollectionProxy->HasPointers()) {
         return *(void **)myCollectionProxy->At(idx);
      } else {
         return myCollectionProxy->At(idx);
      }
   }

   bool IsContiguous(ROOT::Detail::TBranchProxy *proxy) override
   {
      auto cp = GetCP(proxy);
      return IsCPContiguous(*cp);
   }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      auto cp = GetCP(proxy);
      return GetCPValueSize(*cp);
   }
};

// Reader interface for leaf list
// SEE TTreeProxyGenerator.cxx:1319: '//We have a top level raw type'
class TObjectArrayReader : public TVirtualCollectionReader {
private:
   Int_t fBasicTypeSize;

public:
   TObjectArrayReader() : fBasicTypeSize(-1) {}
   ~TObjectArrayReader() override {}
   TVirtualCollectionProxy *GetCP(ROOT::Detail::TBranchProxy *proxy)
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TObjectArrayReader::GetCP()", "Read error in TBranchProxy.");
         return nullptr;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return (TVirtualCollectionProxy *)proxy->GetCollection();
   }
   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return 0;
      return myCollectionProxy->Size();
   }
   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      if (!proxy->Read())
         return nullptr;

      Int_t objectSize;
      void *array = (void *)proxy->GetStart();

      if (fBasicTypeSize == -1) {
         TClass *myClass = proxy->GetClass();
         if (!myClass) {
            Error("TObjectArrayReader::At()", "Cannot get class info from branch proxy.");
            return nullptr;
         }
         objectSize = myClass->GetClassSize();
      } else {
         objectSize = fBasicTypeSize;
      }
      return (void *)((Byte_t *)array + (objectSize * idx));
   }

   void SetBasicTypeSize(Int_t size) { fBasicTypeSize = size; }

   bool IsContiguous(ROOT::Detail::TBranchProxy *) override { return true; }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      auto cp = GetCP(proxy);
      if (cp)
         return GetCPValueSize(*cp);
      else
         return proxy->GetValueSize();
   }
};

template <class BASE>
class TDynamicArrayReader : public BASE {

   // TVirtualSizeReaderImpl and TSizeReaderImpl type-erase the reading of the size leaf.
   class TVirtualSizeReaderImpl {
   public:
      virtual ~TVirtualSizeReaderImpl() = default;
      virtual size_t GetSize() = 0;
   };

   template <typename T>
   class TSizeReaderImpl final : public TVirtualSizeReaderImpl {
      TTreeReaderValue<T> fSizeReader;

   public:
      TSizeReaderImpl(TTreeReader &r, const char *leafName) : fSizeReader(r, leafName) {}
      size_t GetSize() final { return *fSizeReader; }
   };

   std::unique_ptr<TVirtualSizeReaderImpl> fSizeReader;

public:
   template <class... ARGS>
   TDynamicArrayReader(TTreeReader *treeReader, const char *leafName, ARGS &&...args)
      : BASE(std::forward<ARGS>(args)...)
   {
      std::string foundLeafName = leafName;
      TLeaf *sizeLeaf = treeReader->GetTree()->FindLeaf(foundLeafName.c_str());

      if (!sizeLeaf) {
         // leafName might be "top.currentParent.N". But "N" might really be "top.N"!
         // Strip parents until we find the leaf.
         std::string leafNameNoParent = leafName;
         std::string parent;
         auto posLastDot = leafNameNoParent.rfind('.');
         if (posLastDot != leafNameNoParent.npos) {
            parent = leafNameNoParent.substr(0, posLastDot);
            leafNameNoParent.erase(0, posLastDot + 1);
         }

         do {
            if (!sizeLeaf && !parent.empty()) {
               auto posLastDotParent = parent.rfind('.');
               if (posLastDotParent != parent.npos)
                  parent = parent.substr(0, posLastDot);
               else
                  parent.clear();
            }

            foundLeafName = parent;
            if (!parent.empty())
               foundLeafName += ".";
            foundLeafName += leafNameNoParent;
            sizeLeaf = treeReader->GetTree()->FindLeaf(foundLeafName.c_str());
         } while (!sizeLeaf && !parent.empty());
      }

      if (!sizeLeaf) {
         Error("TDynamicArrayReader ", "Cannot find leaf count for %s or any parent branch!", leafName);
         return;
      }

      const std::string leafType = sizeLeaf->GetTypeName();
      if (leafType == "Int_t") {
         fSizeReader.reset(new TSizeReaderImpl<Int_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "UInt_t") {
         fSizeReader.reset(new TSizeReaderImpl<UInt_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "Short_t") {
         fSizeReader.reset(new TSizeReaderImpl<Short_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "UShort_t") {
         fSizeReader.reset(new TSizeReaderImpl<UShort_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "Long_t") {
         fSizeReader.reset(new TSizeReaderImpl<Long_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "ULong_t") {
         fSizeReader.reset(new TSizeReaderImpl<ULong_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "Long64_t") {
         fSizeReader.reset(new TSizeReaderImpl<Long64_t>(*treeReader, foundLeafName.c_str()));
      } else if (leafType == "ULong64_t") {
         fSizeReader.reset(new TSizeReaderImpl<ULong64_t>(*treeReader, foundLeafName.c_str()));
      } else {
         Error("TDynamicArrayReader ",
               "Unsupported size type for leaf %s. Supported types are int, short int, long int, long long int and "
               "their unsigned counterparts.",
               leafName);
      }
   }

   size_t GetSize(ROOT::Detail::TBranchProxy * /*proxy*/) override { return fSizeReader->GetSize(); }
};

class TArrayParameterSizeReader : public TDynamicArrayReader<TObjectArrayReader> {
public:
   TArrayParameterSizeReader(TTreeReader *treeReader, const char *branchName)
      : TDynamicArrayReader<TObjectArrayReader>(treeReader, branchName)
   {
   }
};

// Reader interface for fixed size arrays
class TArrayFixedSizeReader : public TObjectArrayReader {
private:
   Int_t fSize;

public:
   TArrayFixedSizeReader(Int_t sizeArg) : fSize(sizeArg) {}

   size_t GetSize(ROOT::Detail::TBranchProxy * /*proxy*/) override { return fSize; }
};

class TBasicTypeArrayReader final : public TVirtualCollectionReader {
public:
   ~TBasicTypeArrayReader() override {}

   TVirtualCollectionProxy *GetCP(ROOT::Detail::TBranchProxy *proxy)
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TBasicTypeArrayReader::GetCP()", "Read error in TBranchProxy.");
         return nullptr;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return (TVirtualCollectionProxy *)proxy->GetCollection();
   }

   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return 0;
      return myCollectionProxy->Size();
   }

   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      TVirtualCollectionProxy *myCollectionProxy = GetCP(proxy);
      if (!myCollectionProxy)
         return nullptr;
      return (Byte_t *)myCollectionProxy->At(idx) + proxy->GetOffset();
   }

   bool IsContiguous(ROOT::Detail::TBranchProxy *) override { return false; }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      if (!proxy->Read()) {
         fReadStatus = TTreeReaderValueBase::kReadError;
         if (!proxy->GetSuppressErrorsForMissingBranch())
            Error("TBasicTypeArrayReader::GetValueSize()", "Read error in TBranchProxy.");
         return 0;
      }
      fReadStatus = TTreeReaderValueBase::kReadSuccess;
      return proxy->GetValueSize();
   }
};

class TBasicTypeClonesReader final : public TClonesReader {
private:
   Int_t fOffset;

public:
   TBasicTypeClonesReader(Int_t offsetArg) : fOffset(offsetArg) {}

   ~TBasicTypeClonesReader() final = default;
   TBasicTypeClonesReader(const TBasicTypeClonesReader &) = delete;
   TBasicTypeClonesReader &operator=(const TBasicTypeClonesReader &) = delete;
   TBasicTypeClonesReader(TBasicTypeClonesReader &&) = delete;
   TBasicTypeClonesReader &operator=(TBasicTypeClonesReader &&) = delete;

   void *At(ROOT::Detail::TBranchProxy *proxy, size_t idx) override
   {
      TClonesArray *myClonesArray = GetCA(proxy);
      if (!myClonesArray)
         return nullptr;
      return (Byte_t *)myClonesArray->At(idx) + fOffset;
   }
};

class TLeafReader : public TVirtualCollectionReader {
private:
   TTreeReaderValueBase *fValueReader;
   Int_t fElementSize;

public:
   TLeafReader(TTreeReaderValueBase *valueReaderArg) : fValueReader(valueReaderArg), fElementSize(-1) {}

   size_t GetSize(ROOT::Detail::TBranchProxy * /*proxy*/) override
   {
      TLeaf *myLeaf = fValueReader->GetLeaf();
      return myLeaf ? myLeaf->GetLen() : 0; // Error will be printed by GetLeaf
   }

   void *At(ROOT::Detail::TBranchProxy * /*proxy*/, size_t idx) override
   {
      ProxyRead();
      void *address = fValueReader->GetAddress();
      if (fElementSize == -1) {
         TLeaf *myLeaf = fValueReader->GetLeaf();
         if (!myLeaf)
            return nullptr; // Error will be printed by GetLeaf
         fElementSize = myLeaf->GetLenType();
      }
      return (Byte_t *)address + (fElementSize * idx);
   }

   bool IsContiguous(ROOT::Detail::TBranchProxy *) override { return true; }

   std::size_t GetValueSize(ROOT::Detail::TBranchProxy *) override
   {
      auto *leaf = fValueReader->GetLeaf();
      return leaf ? leaf->GetLenType() : 0;
   }

protected:
   void ProxyRead() { fValueReader->ProxyRead(); }
};

class TLeafParameterSizeReader : public TDynamicArrayReader<TLeafReader> {
public:
   TLeafParameterSizeReader(TTreeReader *treeReader, const char *leafName, TTreeReaderValueBase *valueReaderArg)
      : TDynamicArrayReader<TLeafReader>(treeReader, leafName, valueReaderArg)
   {
   }

   size_t GetSize(ROOT::Detail::TBranchProxy *proxy) override
   {
      ProxyRead();
      return TDynamicArrayReader<TLeafReader>::GetSize(proxy);
   }
};
} // namespace

ClassImp(TTreeReaderArrayBase);

////////////////////////////////////////////////////////////////////////////////
/// Create the proxy object for our branch.

void ROOT::Internal::TTreeReaderArrayBase::CreateProxy()
{
   if (fProxy) {
      return;
   }

   fSetupStatus = kSetupInternalError; // Fallback; set to something concrete below.
   if (!fTreeReader) {
      Error("TTreeReaderArrayBase::CreateProxy()", "TTreeReader object not set / available for branch %s!",
            fBranchName.Data());
      fSetupStatus = kSetupTreeDestructed;
      return;
   }
   if (!fDict) {
      TBranch *br = fTreeReader->GetTree()->GetBranch(fBranchName);
      const char *brDataType = "{UNDETERMINED}";
      if (br) {
         TDictionary *dictUnused = nullptr;
         brDataType = GetBranchDataType(br, dictUnused, fDict);
      }
      Error("TTreeReaderArrayBase::CreateProxy()",
            "The template argument type T of %s accessing branch %s (which contains data of type %s) is not known to "
            "ROOT. You will need to create a dictionary for it.",
            GetDerivedTypeName(), fBranchName.Data(), brDataType);
      fSetupStatus = kSetupMissingDictionary;
      return;
   }

   // Access a branch's collection content (not the collection itself)
   // through a proxy.
   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.

   // Tell the branch proxy to suppress the errors for missing branch if this
   // branch name is found in the list of suppressions
   const bool suppressErrorsForThisBranch = (fTreeReader->fSuppressErrorsForMissingBranches.find(fBranchName.Data()) !=
                                             fTreeReader->fSuppressErrorsForMissingBranches.cend());

   TDictionary *branchActualType = nullptr;
   TBranch *branch = nullptr;
   TLeaf *myLeaf = nullptr;
   if (!GetBranchAndLeaf(branch, myLeaf, branchActualType, suppressErrorsForThisBranch))
      return;

   if (!fDict) {
      Error("TTreeReaderArrayBase::CreateProxy()", "No dictionary for branch %s.", fBranchName.Data());
      return;
   }

   TNamedBranchProxy *namedProxy = fTreeReader->FindProxy(fBranchName);
   if (namedProxy) {
      if (namedProxy->GetContentDict() == fDict) {
         fSetupStatus = kSetupMatch;
         fProxy = namedProxy->GetProxy();
         SetImpl(branch, myLeaf);
         return;
      }

      // Update named proxy's dictionary
      if (!namedProxy->GetContentDict()) {
         namedProxy->SetContentDict(fDict);
         fProxy = namedProxy->GetProxy();
         if (fProxy)
            fSetupStatus = kSetupMatch;
      } else {
         Error("TTreeReaderArrayBase::CreateProxy()", "Type ambiguity (want %s, have %s) for branch %s.",
               fDict->GetName(), namedProxy->GetContentDict()->GetName(), fBranchName.Data());
      }
   } else {
      TString membername;

      bool isTopLevel = branch->GetMother() == branch;
      if (!isTopLevel) {
         membername = strrchr(branch->GetName(), '.');
         if (membername.IsNull()) {
            membername = branch->GetName();
         }
      }
      auto *director = fTreeReader->fDirector.get();
      // Determine if the branch is actually in a Friend TTree and if so which.
      if (branch->GetTree() != fTreeReader->GetTree()->GetTree()) {
         // It is in a friend, let's find the 'index' in the list of friend ...
         std::optional<std::size_t> index;
         std::size_t current{};
         auto &&friends = fTreeReader->GetTree()->GetTree()->GetListOfFriends();
         for (auto fe : TRangeDynCast<TFriendElement>(friends)) {
            if (branch->GetTree() == fe->GetTree()) {
               index = current;
               break;
            }
            ++current;
         }
         if (!index.has_value()) {
            Error("TTreeReaderArrayBase::CreateProxy()",
                  "The branch %s is contained in a Friend TTree that is not directly attached to the main.\n"
                  "This is not yet supported by TTreeReader.",
                  fBranchName.Data());
            return;
         }

         auto &&friendProxy = fTreeReader->AddFriendProxy(index.value());
         director = friendProxy.GetDirector();
      }
      fTreeReader->AddProxy(
         std::make_unique<TNamedBranchProxy>(director, branch, fBranchName, membername, suppressErrorsForThisBranch));

      namedProxy = fTreeReader->FindProxy(fBranchName);
      fProxy = namedProxy->GetProxy();
      if (fProxy)
         fSetupStatus = kSetupMatch;
      else
         fSetupStatus = kSetupMismatch;
   }

   if (!myLeaf) {
      TString branchActualTypeName;
      const char *nonCollTypeName = GetBranchContentDataType(branch, branchActualTypeName, branchActualType);
      if (nonCollTypeName) {
         Error("TTreeReaderArrayBase::CreateContentProxy()",
               "The branch %s contains data of type %s, which should be accessed through a TTreeReaderValue< %s >.",
               fBranchName.Data(), nonCollTypeName, nonCollTypeName);
         if (fSetupStatus == kSetupInternalError)
            fSetupStatus = kSetupNotACollection;
         fProxy = nullptr;
         return;
      }
      if (!branchActualType) {
         if (branchActualTypeName.IsNull()) {
            Error("TTreeReaderArrayBase::CreateContentProxy()",
                  "Cannot determine the type contained in the collection of branch %s. That's weird - please report!",
                  fBranchName.Data());
         } else {
            Error("TTreeReaderArrayBase::CreateContentProxy()",
                  "The branch %s contains data of type %s, which does not have a dictionary.", fBranchName.Data(),
                  branchActualTypeName.Data());
            if (fSetupStatus == kSetupInternalError)
               fSetupStatus = kSetupMissingDictionary;
         }
         fProxy = nullptr;
         return;
      }

      auto matchingDataType = [](TDictionary *left, TDictionary *right) -> bool {
         if (left == right)
            return true;
         if (!left || !right)
            return false;
         auto left_datatype = dynamic_cast<TDataType *>(left);
         auto right_datatype = dynamic_cast<TDataType *>(right);
         auto left_enum = dynamic_cast<TEnum *>(left);
         auto right_enum = dynamic_cast<TEnum *>(right);

         if ((left_datatype && left_datatype->GetType() == kInt_t && right_enum) ||
             (right_datatype && right_datatype->GetType() == kInt_t && left_enum))
            return true;
         if ((left_datatype && right_enum && left_datatype->GetType() == right_enum->GetUnderlyingType()) ||
             (right_datatype && left_enum && right_datatype->GetType() == left_enum->GetUnderlyingType()))
            return true;
         if (!left_datatype || !right_datatype)
            return false;
         auto l = left_datatype->GetType();
         auto r = right_datatype->GetType();
         if (l > 0 && l == r)
            return true;
         else
            return ((l == kDouble32_t && r == kDouble_t) || (l == kDouble_t && r == kDouble32_t) ||
                    (l == kFloat16_t && r == kFloat_t) || (l == kFloat_t && r == kFloat16_t));
      };

      if (!matchingDataType(fDict, branchActualType)) {
         Error("TTreeReaderArrayBase::CreateContentProxy()",
               "The branch %s contains data of type %s. It cannot be accessed by a TTreeReaderArray<%s>",
               fBranchName.Data(), branchActualType->GetName(), fDict->GetName());
         if (fSetupStatus == kSetupInternalError || fSetupStatus >= 0)
            fSetupStatus = kSetupMismatch;

         // Update named proxy's dictionary
         if (!namedProxy->GetContentDict()) {
            namedProxy->SetContentDict(fDict);
         }

         // fProxy = 0;
         // return;
      }
   }

   SetImpl(branch, myLeaf);
}

////////////////////////////////////////////////////////////////////////////////
/// Determine the branch / leaf and its type; reset fProxy / fSetupStatus on error.

bool ROOT::Internal::TTreeReaderArrayBase::GetBranchAndLeaf(TBranch *&branch, TLeaf *&myLeaf,
                                                            TDictionary *&branchActualType,
                                                            bool suppressErrorsForMissingBranch)
{
   myLeaf = nullptr;
   branch = fTreeReader->GetTree()->GetBranch(fBranchName);
   if (branch)
      return true;

   if (!fBranchName.Contains(".")) {
      if (!suppressErrorsForMissingBranch) {
         Error("TTreeReaderArrayBase::GetBranchAndLeaf()",
               "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.",
               fBranchName.Data());
      }
      fSetupStatus = kSetupMissingBranch;
      fProxy = nullptr;
      return false;
   }

   TRegexp leafNameExpression("\\.[a-zA-Z0-9_]+$");
   TString leafName(fBranchName(leafNameExpression));
   TString branchName = fBranchName(0, fBranchName.Length() - leafName.Length());
   branch = fTreeReader->GetTree()->GetBranch(branchName);
   if (!branch) {
      if (!suppressErrorsForMissingBranch) {
         Error("TTreeReaderArrayBase::GetBranchAndLeaf()",
               "The tree does not have a branch called %s. You could check with TTree::Print() for available branches.",
               fBranchName.Data());
      }
      fSetupStatus = kSetupMissingBranch;
      fProxy = nullptr;
      return false;
   }

   myLeaf = branch->GetLeaf(TString(leafName(1, leafName.Length())));
   if (!myLeaf) {
      if (!suppressErrorsForMissingBranch) {
         Error("TTreeReaderArrayBase::GetBranchAndLeaf()",
               "The tree does not have a branch, nor a sub-branch called %s. You could check with TTree::Print() for "
               "available branches.",
               fBranchName.Data());
      }
      fSetupStatus = kSetupMissingBranch;
      fProxy = nullptr;
      return false;
   }

   TDictionary *tempDict = TDictionary::GetDictionary(myLeaf->GetTypeName());
   if (!tempDict) {
      Error("TTreeReaderArrayBase::GetBranchAndLeaf()", "Failed to get the dictionary for %s.", myLeaf->GetTypeName());
      fSetupStatus = kSetupMissingDictionary;
      fProxy = nullptr;
      return false;
   }

   if (tempDict->IsA() == TDataType::Class() &&
       TDictionary::GetDictionary(((TDataType *)tempDict)->GetTypeName()) == fDict) {
      // fLeafOffset = myLeaf->GetOffset() / 4;
      branchActualType = fDict;
      fLeaf = myLeaf;
      fBranchName = branchName;
      fLeafName = leafName(1, leafName.Length());
      fHaveLeaf = (fLeafName.Length() > 0);
      fSetupStatus = kSetupMatchLeaf;
   } else {
      Error("TTreeReaderArrayBase::GetBranchAndLeaf()", "Leaf of type %s cannot be read by TTreeReaderValue<%s>.",
            myLeaf->GetTypeName(), fDict->GetName());
      fProxy = nullptr;
      fSetupStatus = kSetupMismatch;
      return false;
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the TVirtualCollectionReader object for our branch.

void ROOT::Internal::TTreeReaderArrayBase::SetImpl(TBranch *branch, TLeaf *myLeaf)
{
   if (fImpl)
      return;

   // Access a branch's collection content (not the collection itself)
   // through a proxy.
   // Search for the branchname, determine what it contains, and wire the
   // TBranchProxy representing it to us so we can access its data.
   // A proxy for branch must not have been created before (i.e. check
   // fProxies before calling this function!)

   if (myLeaf) {
      if (!myLeaf->GetLeafCount()) {
         fImpl = std::make_unique<TLeafReader>(this);
      } else {
         TString leafFullName = myLeaf->GetBranch()->GetName();
         leafFullName += ".";
         leafFullName += myLeaf->GetLeafCount()->GetName();
         fImpl = std::make_unique<TLeafParameterSizeReader>(fTreeReader, leafFullName.Data(), this);
      }
      fSetupStatus = kSetupMatchLeaf;
   } else if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement *branchElement = ((TBranchElement *)branch);

      TStreamerInfo *streamerInfo = branchElement->GetInfo();
      Int_t id = branchElement->GetID();

      if (id >= 0) { // Not root node?
         // Int_t offset = streamerInfo->GetOffsets()[id];
         TStreamerElement *element = (TStreamerElement *)streamerInfo->GetElements()->At(id);
         // bool isPointer = element->IsaPointer();
         // TClass *classPointer = element->GetClassPointer();

         if (fSetupStatus == kSetupInternalError)
            fSetupStatus = kSetupMatch;
         if (element->IsA() == TStreamerSTL::Class()) {
            if (branchElement->GetType() == 31) {
               Error("TTreeReaderArrayBase::SetImpl", "STL Collection nested in a TClonesArray not yet supported");
               fSetupStatus = kSetupInternalError;
               return;
            }
            fImpl = std::make_unique<TSTLReader>();
         } else if (element->IsA() == TStreamerObject::Class()) {
            // fImpl = new TObjectArrayReader(); // BArray[12]

            if (element->GetClass() == TClonesArray::Class()) {
               fImpl = std::make_unique<TClonesReader>();
            } else if (branchElement->GetType() == TBranchElement::kSTLMemberNode) {
               fImpl = std::make_unique<TBasicTypeArrayReader>();
            } else if (branchElement->GetType() == TBranchElement::kClonesMemberNode) {
               // TBasicTypeClonesReader should work for object
               fImpl = std::make_unique<TBasicTypeClonesReader>(element->GetOffset());
            } else {
               fImpl = std::make_unique<TArrayFixedSizeReader>(element->GetArrayLength());
            }
         } else if (element->IsA() == TStreamerLoop::Class()) {
            fImpl =
               std::make_unique<TArrayParameterSizeReader>(fTreeReader, branchElement->GetBranchCount()->GetName());
         } else if (element->IsA() == TStreamerBasicType::Class()) {
            if (branchElement->GetType() == TBranchElement::kSTLMemberNode) {
               fImpl = std::make_unique<TBasicTypeArrayReader>();
            } else if (branchElement->GetType() == TBranchElement::kClonesMemberNode) {
               fImpl = std::make_unique<TBasicTypeClonesReader>(element->GetOffset());
            } else if (fDict->IsA() == TEnum::Class()) {
               fImpl = std::make_unique<TArrayFixedSizeReader>(element->GetArrayLength());
               ((TObjectArrayReader *)fImpl.get())->SetBasicTypeSize(sizeof(Int_t));
            } else {
               fImpl = std::make_unique<TArrayFixedSizeReader>(element->GetArrayLength());
               ((TObjectArrayReader *)fImpl.get())->SetBasicTypeSize(((TDataType *)fDict)->Size());
            }
         } else if (element->IsA() == TStreamerBasicPointer::Class()) {
            fImpl =
               std::make_unique<TArrayParameterSizeReader>(fTreeReader, branchElement->GetBranchCount()->GetName());
            ((TArrayParameterSizeReader *)fImpl.get())->SetBasicTypeSize(((TDataType *)fDict)->Size());
         } else if (element->IsA() == TStreamerBase::Class()) {
            fImpl = std::make_unique<TClonesReader>();
         } else {
            Error("TTreeReaderArrayBase::SetImpl()", "Cannot read branch %s: unhandled streamer element type %s",
                  fBranchName.Data(), element->IsA()->GetName());
            fSetupStatus = kSetupInternalError;
         }
      } else { // We are at root node?
         if (branchElement->GetClass()->GetCollectionProxy()) {
            fImpl = std::make_unique<TCollectionLessSTLReader>(branchElement->GetClass()->GetCollectionProxy());
         }
      }
   } else if (branch->IsA() == TBranch::Class()) {
      auto topLeaf = branch->GetLeaf(branch->GetName());
      if (!topLeaf) {
         Error("TTreeReaderArrayBase::SetImpl", "Failed to get the top leaf from the branch");
         fSetupStatus = kSetupMissingBranch;
         return;
      }
      // We could have used GetLeafCounter, but it does not work well with Double32_t and Float16_t: ROOT-10149
      auto sizeLeaf = topLeaf->GetLeafCount();
      if (fSetupStatus == kSetupInternalError)
         fSetupStatus = kSetupMatch;
      if (!sizeLeaf) {
         fImpl = std::make_unique<TArrayFixedSizeReader>(topLeaf->GetLenStatic());
      } else {
         fImpl = std::make_unique<TArrayParameterSizeReader>(fTreeReader, sizeLeaf->GetName());
      }
      ((TObjectArrayReader *)fImpl.get())->SetBasicTypeSize(((TDataType *)fDict)->Size());
   } else if (branch->IsA() == TBranchClones::Class()) {
      Error("TTreeReaderArrayBase::SetImpl", "Support for branches of type TBranchClones not implemented");
      fSetupStatus = kSetupInternalError;
   } else if (branch->IsA() == TBranchObject::Class()) {
      Error("TTreeReaderArrayBase::SetImpl", "Support for branches of type TBranchObject not implemented");
      fSetupStatus = kSetupInternalError;
   } else if (branch->IsA() == TBranchSTL::Class()) {
      Error("TTreeReaderArrayBase::SetImpl", "Support for branches of type TBranchSTL not implemented");
      fImpl = std::make_unique<TSTLReader>();
      fSetupStatus = kSetupInternalError;
   } else if (branch->IsA() == TBranchRef::Class()) {
      Error("TTreeReaderArrayBase::SetImpl", "Support for branches of type TBranchRef not implemented");
      fSetupStatus = kSetupInternalError;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Access a branch's collection content (not the collection itself)
/// through a proxy.
/// Retrieve the type of data contained in the collection stored by branch;
/// put its dictionary into dict, If there is no dictionary, put its type
/// name into contentTypeName.
/// The contentTypeName is set to NULL if the branch does not
/// contain a collection; in that case, the type of the branch is returned.
/// In all other cases, NULL is returned.

const char *ROOT::Internal::TTreeReaderArrayBase::GetBranchContentDataType(TBranch *branch, TString &contentTypeName,
                                                                           TDictionary *&dict)
{
   dict = nullptr;
   contentTypeName = "";
   if (branch->IsA() == TBranchElement::Class()) {
      TBranchElement *brElement = (TBranchElement *)branch;
      if (brElement->GetType() == 4 || brElement->GetType() == 3) {
         TVirtualCollectionProxy *collProxy = brElement->GetCollectionProxy();
         if (collProxy) {
            TClass *myClass = collProxy->GetValueClass();
            if (!myClass) {
               Error("TTreeReaderArrayBase::GetBranchContentDataType()", "Could not get value class.");
               return nullptr;
            }
            dict = TDictionary::GetDictionary(myClass->GetName());
            if (!dict)
               dict = TDataType::GetDataType(collProxy->GetType());
         }
         if (!dict) {
            // We don't know the dictionary, thus we need the content's type name.
            // Determine it.
            if (brElement->GetType() == 3) {
               contentTypeName = brElement->GetClonesName();
               dict = TDictionary::GetDictionary(brElement->GetClonesName());
               return nullptr;
            }
            // STL:
            TClassEdit::TSplitType splitType(brElement->GetClassName());
            int isSTLCont = splitType.IsSTLCont();
            if (!isSTLCont) {
               Error("TTreeReaderArrayBase::GetBranchContentDataType()",
                     "Cannot determine STL collection type of %s stored in branch %s", brElement->GetClassName(),
                     branch->GetName());
               return brElement->GetClassName();
            }
            bool isMap = isSTLCont == ROOT::kSTLmap || isSTLCont == ROOT::kSTLmultimap;
            if (isMap)
               contentTypeName = "std::pair< ";
            contentTypeName += splitType.fElements[1];
            if (isMap) {
               contentTypeName += splitType.fElements[2];
               contentTypeName += " >";
            }
            return nullptr;
         }
         return nullptr;
      } else if (brElement->GetType() == 31 || brElement->GetType() == 41) {
         // it's a member, extract from GetClass()'s streamer info
         TClass *clData = nullptr;
         EDataType dtData = kOther_t;
         int ExpectedTypeRet = brElement->GetExpectedType(clData, dtData);
         if (ExpectedTypeRet == 0) {
            dict = clData;
            if (!dict) {
               if (dtData == kFloat16_t) {
                  dtData = kFloat_t;
               }
               if (dtData == kDouble32_t) {
                  dtData = kDouble_t;
               }
               dict = TDataType::GetDataType(dtData);
            }
            if (!dict) {
               Error("TTreeReaderArrayBase::GetBranchContentDataType()",
                     "The branch %s contains a data type %d for which the dictionary cannot be retrieved.",
                     branch->GetName(), (int)dtData);
               contentTypeName = TDataType::GetTypeName(dtData);
               return nullptr;
            }
            return nullptr;
         } else if (ExpectedTypeRet == 1) {
            int brID = brElement->GetID();
            if (brID == -1) {
               // top
               Error("TTreeReaderArrayBase::GetBranchContentDataType()",
                     "The branch %s contains data of type %s for which the dictionary does not exist. It's needed.",
                     branch->GetName(), brElement->GetClassName());
               contentTypeName = brElement->GetClassName();
               return nullptr;
            }
            // Either the data type name doesn't have an EDataType entry
            // or the streamer info doesn't have a TClass* attached.
            TStreamerElement *element = (TStreamerElement *)brElement->GetInfo()->GetElement(brID);
            contentTypeName = element->GetTypeName();
            return nullptr;
         }
         /* else (ExpectedTypeRet == 2)*/
         // The streamer info entry cannot be found.
         // TBranchElement::GetExpectedType() has already complained.
         return "{CANNOT DETERMINE TBranchElement DATA TYPE}";
      } else if (brElement->GetType() == TBranchElement::kLeafNode) {
         TStreamerInfo *streamerInfo = brElement->GetInfo();
         Int_t id = brElement->GetID();

         if (id >= 0) {
            TStreamerElement *element = (TStreamerElement *)streamerInfo->GetElements()->At(id);

            if (element->IsA() == TStreamerSTL::Class()) {
               TClass *myClass = brElement->GetCurrentClass();
               if (!myClass) {
                  Error("TTreeReaderArrayBase::GetBranchDataType()", "Could not get class from branch element.");
                  return nullptr;
               }
               TVirtualCollectionProxy *myCollectionProxy = myClass->GetCollectionProxy();
               if (!myCollectionProxy) {
                  Error("TTreeReaderArrayBase::GetBranchDataType()", "Could not get collection proxy from STL class");
                  return nullptr;
               }
               // Try getting the contained class
               dict = myCollectionProxy->GetValueClass();
               // If it fails, try to get the contained type as a primitive type
               if (!dict)
                  dict = TDataType::GetDataType(myCollectionProxy->GetType());
               if (!dict) {
                  Error("TTreeReaderArrayBase::GetBranchDataType()", "Could not get valueClass from collectionProxy.");
                  return nullptr;
               }
               contentTypeName = dict->GetName();
               return nullptr;
            } else if (element->IsA() == TStreamerObject::Class() && !strcmp(element->GetTypeName(), "TClonesArray")) {
               if (!fProxy->Setup() || !fProxy->Read()) {
                  Error("TTreeReaderArrayBase::GetBranchContentDataType()",
                        "Failed to get type from proxy, unable to check type");
                  contentTypeName = "UNKNOWN";
                  dict = nullptr;
                  return contentTypeName;
               }
               TClonesArray *myArray = (TClonesArray *)fProxy->GetWhere();
               dict = myArray->GetClass();
               contentTypeName = dict->GetName();
               return nullptr;
            } else {
               dict = brElement->GetCurrentClass();
               if (!dict) {
                  TDictionary *myDataType = TDictionary::GetDictionary(brElement->GetTypeName());
                  dict = TDataType::GetDataType((EDataType)((TDataType *)myDataType)->GetType());
               }
               contentTypeName = brElement->GetTypeName();
               return nullptr;
            }
         }
         if (brElement->GetCurrentClass() == TClonesArray::Class()) {
            contentTypeName = "TClonesArray";
            Warning("TTreeReaderArrayBase::GetBranchContentDataType()",
                    "Not able to check type correctness, ignoring check");
            dict = fDict;
            fSetupStatus = kSetupNoCheck;
         } else if (!dict && (branch->GetSplitLevel() == 0 || brElement->GetClass()->GetCollectionProxy())) {
            // Try getting the contained class
            dict = brElement->GetClass()->GetCollectionProxy()->GetValueClass();
            // If it fails, try to get the contained type as a primitive type
            if (!dict)
               dict = TDataType::GetDataType(brElement->GetClass()->GetCollectionProxy()->GetType());
            if (dict)
               contentTypeName = dict->GetName();
            return nullptr;
         } else if (!dict) {
            dict = brElement->GetClass();
            contentTypeName = dict->GetName();
            return nullptr;
         }

         return nullptr;
      }
      return nullptr;
   } else if (branch->IsA() == TBranch::Class() || branch->IsA() == TBranchObject::Class() ||
              branch->IsA() == TBranchSTL::Class()) {
      const char *dataTypeName = branch->GetClassName();
      if ((!dataTypeName || !dataTypeName[0]) && branch->IsA() == TBranch::Class()) {
         auto myLeaf = branch->GetLeaf(branch->GetName());
         if (myLeaf) {
            auto myDataType = TDictionary::GetDictionary(myLeaf->GetTypeName());
            if (myDataType && myDataType->IsA() == TDataType::Class()) {
               auto typeEnumConstant = EDataType(((TDataType *)myDataType)->GetType());
               // We need to consider Double32_t and Float16_t as dounle and float respectively
               // since this is the type the user uses to instantiate the TTreeReaderArray template.
               if (typeEnumConstant == kDouble32_t)
                  typeEnumConstant = kDouble_t;
               else if (typeEnumConstant == kFloat16_t)
                  typeEnumConstant = kFloat_t;
               dict = TDataType::GetDataType(typeEnumConstant);
               contentTypeName = myLeaf->GetTypeName();
               return nullptr;
            }
         }

         // leaflist. Can't represent.
         Error("TTreeReaderArrayBase::GetBranchContentDataType()",
               "The branch %s was created using a leaf list and cannot be represented as a C++ type. Please access one "
               "of its siblings using a TTreeReaderArray:",
               branch->GetName());
         TIter iLeaves(branch->GetListOfLeaves());
         TLeaf *leaf = nullptr;
         while ((leaf = (TLeaf *)iLeaves())) {
            Error("TTreeReaderArrayBase::GetBranchContentDataType()", "   %s.%s", branch->GetName(), leaf->GetName());
         }
         return nullptr;
      }
      if (dataTypeName)
         dict = TDictionary::GetDictionary(dataTypeName);
      if (branch->IsA() == TBranchSTL::Class()) {
         Warning("TTreeReaderArrayBase::GetBranchContentDataType()",
                 "Not able to check type correctness, ignoring check");
         dict = fDict;
         fSetupStatus = kSetupNoCheck;
         return nullptr;
      }
      return dataTypeName;
   } else if (branch->IsA() == TBranchClones::Class()) {
      dict = TClonesArray::Class();
      return "TClonesArray";
   } else if (branch->IsA() == TBranchRef::Class()) {
      // Can't represent.
      Error("TTreeReaderArrayBase::GetBranchContentDataType()",
            "The branch %s is a TBranchRef and cannot be represented as a C++ type.", branch->GetName());
      return nullptr;
   } else {
      Error("TTreeReaderArrayBase::GetBranchContentDataType()",
            "The branch %s is of type %s - something that is not handled yet.", branch->GetName(),
            branch->IsA()->GetName());
      return nullptr;
   }

   return nullptr;
}
